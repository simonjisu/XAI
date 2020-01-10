from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchxai.trainer import XaiTrainer
from .models.plaincnn import CnnMnist
from .models.resnet import ResNetMnist, ResNetMnistCBAM, ResNetCifar10, ResNetCifar10CBAM
from collections import OrderedDict


class ModelTranier(XaiTrainer):
    def __init__(self):
        """
        contains default
        args: `self.attr_dict`
        function: `check_path_exist`
        """
        super(ModelTranier, self).__init__()
        self.dataset_dict = {
            "mnist": datasets.MNIST,
            "cifar10": datasets.CIFAR10
        } 
        self.transform_dict = {
            "mnist": {
                "train": transforms.ToTensor(),
                "test": transforms.ToTensor(),
            },
            "cifar10": {
                "train": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                    ]),
                "test": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                    ])
            }
        }
        
        self.model_dict = {
            "mnist": {
                "cnn": CnnMnist,
                "resnet": ResNetMnist,
                "resnetcbam": ResNetMnistCBAM
            },
            "cifar10": {
                "cnn": None,
                "resnet": ResNetCifar10, 
                "resnetcbam": ResNetCifar10CBAM
            }
        }

        # give some arguments to attribution models
        self.kwargs_packs = {
            "mnist": {
                "gradcam": {
                    "cnn": dict(layers_name=None, norm_mode=1),
                    "resnet": dict(layers_name="relu_last", norm_mode=1),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=1)
                },
                "guidedgrad": {
                    "cnn": dict(module_name="convs", act=nn.ReLU),
                    "resnet": dict(module_name=["resnet_layers", "relu"], act=nn.ReLU),
                    "resnetcbam": dict(module_name=["resnet_layers", "relu"], act=nn.ReLU)
                },
                "relavance": {
                    "cnn": dict(use_rho=False)
                },
                "deconv": {
                    "cnn": dict(module_name="convs")
                }, 
                "vanillagrad": None, 
                "inputgrad": None
            },
            "cifar10": {
                "gradcam": {
                    "resnet": dict(layers_name="relu_last", norm_mode=3),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=3)
                },
                "guidedgrad": {
                    "resnet": dict(module_name=["resnet_layers", "relu"], act=nn.ReLU),
                    "resnetcbam": dict(module_name=["resnet_layers", "relu"], act=nn.ReLU)
                },
                "vanillagrad": None, 
                "inputgrad": None
            }
            
        }

        self.loss_fn_dict = {
            
        }

    def train(self, model, train_loader, optimizer, loss_function, device):
        train_loss = 0
        model.train()
        for datas, targets in train_loader:
            datas, targets = datas.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(datas)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
                    
        return train_loss

    def test(self, model, test_loader, device):
        model.eval()
        test_loss = 0
        corrects = 0
        with torch.no_grad():
            for datas, targets in test_loader:
                datas, targets = datas.to(device), targets.to(device)
                outputs = model(datas)
                test_loss += F.cross_entropy(outputs, targets, reduction="sum").item()
                preds = outputs.argmax(dim=1, keepdim=True)
                corrects += preds.eq(targets.view_as(preds)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_acc = 100*(corrects / len(test_loader.dataset))

        return test_loss, test_acc

    def main_train(self, model, train_loader, test_loader, n_step, sv_path, device):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        best_acc = 0.0
        for step in range(n_step):
            train_loss = self.train(model, train_loader, optimizer, loss_function, device)
            test_loss, test_acc = self.test(model, test_loader, device)
            print(f"[Step] {step+1}/{n_step}")
            print(f"[Train] Average loss: {train_loss:.4f}")
            print(f"[Test] Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), sv_path)
                print("[Alert] best model saved")
            print("----"*10)
        return best_acc

    def build_dataset(self, args, shuffle=True):
        train_dataset = self.dataset_dict[args.data_type](
            root=args.data_path,                                
            train=True,
            transform=self.transform_dict[args.data_type]["train"],
            download=args.download)
        test_dataset = self.dataset_dict[args.data_type](
            root=args.data_path, 
            train=False,
            transform=self.transform_dict[args.data_type]["test"],
            download=args.download)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, 
            shuffle=shuffle)
        return train_dataset, test_dataset, train_loader, test_loader

    def init_masks_dict(self, args, train_dataset, test_dataset):
        if args.data_type == "mnist":
            B_train, H, W = train_dataset.data.shape
            C = 1
        elif args.data_type == "cifar10":
            B_train, H, W, C = train_dataset.data.shape
        B_test, *_ = test_dataset.data.shape
        
        masks_dict = {
            "train": OrderedDict(), 
            "train-mask": OrderedDict(), 
            "test": OrderedDict(),
            "test-mask": OrderedDict()
        }                  
        masks_dict["train-mask"][0.0] = torch.zeros(B_train, C, H, W).bool()
        masks_dict["test-mask"][0.0] = torch.zeros(B_test, C, H, W).bool()
        return masks_dict

    def check_if_model_supported(self, args):
        # relavance method is restricted to use custom models
        pass
    
    def convert_scale(self, args, outputs, C):
        """
        reference: https://en.wikipedia.org/wiki/Grayscale
        methods:
        [1] rec601
        In the Y'UV and Y'IQ models used by PAL and NTSC, 
        the rec601 luma component is computed as
        Y = 0.299*R + 0.587*G + 0.114*B
        [2] itu-r_bt.707
        The ITU-R BT.709 standard used for HDTV developed by the ATSC 
        uses different color coefficients, computing the luma component as
        Y = 0.2126*R + 0.7152*G + 0.0722*B
        [3] itu-r_bt.2100
        The ITU-R BT.2100 standard for HDR television uses yet different 
        coefficients, computing the luma component as
        Y = 0.2627*R + 0.6780*G + 0.0593*B
        """
        if outputs.size(1) == 1:
            return outputs, C        
        else:
            method_dict = {
                "rec601": lambda r, g, b: 0.299*r + 0.587*g + 0.114*b,
                "itu-r_bt.707": lambda r, g, b: 0.2126*r + 0.7152*g + 0.0722*b,
                "iut-r_bt.2100": lambda r, g, b: 0.2627*r + 0.6780*g + 0.0593*b,
            }
            if args.reduce_color_dim is not None:
                f = method_dict[args.reduce_color_dim]
                outputs = f(outputs[:, 0, :, :], outputs[:, 1, :, :], outputs[:, 2, :, :])
                if len(outputs.size()) == 3:
                    outputs = outputs.unsqueeze(1)
                C = 1
                return outputs, C
            else:
                assert False, "args.reduce_color_dim is not defined."

    def get_new_data(self, args, del_p, typ):
        """
        returns masked data

        args: argparse object
        del_p: delete percentages
        typ: whether is train or test

        if `args.reduce_color_dim` option is not None:
            recude all channel dimention to 1
        """
        datas = self.masks_dict[typ][del_p]
        masks = self.masks_dict[typ+"-mask"][del_p]
        new_datas = datas.masked_fill(masks, 0.0)
        # B, C, H, W = datas.size()
        if args.data_type.lower() == "mnist":
            # datashape: (B, H, W)
            return new_datas.squeeze()
        elif args.data_type.lower() == "cifar10":
            # datashape: (B, H, W, C)
            return new_datas.permute(0, 2, 3, 1).numpy()

    def get_masks(self, args, data_loader, attr_model, del_p, typ):
        """
        Get masks to delete and corresponding datas
        """
        def calculate_masks(args, outputs, del_p):
            B, C, H, W = outputs.size()
            outputs, C = self.convert_scale(args, outputs, C)
            reshaped_outputs = outputs.view(B, C, -1)
            vals, _ = reshaped_outputs.sort(-1)
            # decide how many pixels to delete
            del_n_idx = torch.LongTensor([int(del_p * H * W)])  
            del_vals = vals.index_select(-1, del_n_idx)
            del_masks = (reshaped_outputs <= del_vals).view(B, C, H, W)
            return del_masks

        temp = []
        temp_masks = []
        for datas, targets in tqdm(data_loader, 
                                   desc=f"- [{typ}]deleting {del_p*100}% datas by attributions", 
                                   total=len(data_loader)):
            temp.append(datas)
            outputs = attr_model.get_attribution(datas, targets).detach()
            masks = calculate_masks(args, outputs, del_p)
            temp_masks.append(masks)

        all_datas = torch.cat(temp, dim=0)
        all_masks = torch.cat(temp_masks, dim=0)

        self.masks_dict[typ][del_p] = all_datas
        previous_del_p = list(self.masks_dict[typ+"-mask"])[-1]
        self.masks_dict[typ+"-mask"][del_p] = all_masks + self.masks_dict[typ+"-mask"][previous_del_p]

    def create_attr_model(self, model_class, attr_class, load_path, **kwargs):
        """
        recreate the model & load its weights. after this create an attribution model
        """
        model = model_class()
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        attr_model = attr_class(model, **kwargs)
        return model, attr_model

    def evaluation(self, args, attr_model, del_p):
        train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=False)
        self.get_masks(args, train_loader, attr_model, del_p, typ="train")
        self.get_masks(args, test_loader, attr_model, del_p, typ="test")

        train_dataset.data = self.get_new_data(args, del_p, typ="train")
        test_dataset.data = self.get_new_data(args, del_p, typ="test")
        # recreate new data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        return train_loader, test_loader

    def roar(self, args, model_class, attr_class, del_p, sv_path, load_path, device, **kwargs):
        # after deletion retrain
        model, attr_model = self.create_attr_model(model_class, attr_class, load_path, **kwargs)
        train_loader, test_loader = self.evaluation(args, attr_model, del_p)
        # start to retrain model
        print("[Alert] Start Retraining")
        model = model.to(device)
        best_acc = self.main_train(model, train_loader, test_loader, args.n_step, sv_path, device)
        
        return best_acc

    def selectivity(self, args, model_class, attr_class, del_p, sv_path, load_path, device):
        raise NotImplementedError

    def record_result(self, record_path, create=False, **kwargs):
        if create:
            self.check_path_exist(record_path, directory=False)
            with record_path.open(mode="w", encoding="utf-8") as f:
                f.write("| model_type | attr_method_type | deleted_percentages | best_acc |\n")
                f.write("|--|--|--|--|\n")
        else:
            model_type = kwargs["model_type"]
            attr_type = kwargs["attr_type"]
            del_p = kwargs["del_p"]
            best_acc= kwargs["best_acc"]
            with record_path.open(mode="a", encoding="utf-8") as f:
                f.write(f"|{model_type}|{attr_type}|{del_p*100:}%|{best_acc:.2f}%|\n")

    def get_kwargs_to_attr_model(self, data_type, m_type, a_type):
        pre_kwargs = self.kwargs_packs[data_type].get(a_type)
        if pre_kwargs is not None:
            kwargs = pre_kwargs.get(m_type)
        else:
            kwargs = {"<none>": None}
        return kwargs

    def main(self, args):
        """
        `kwargs` will deliver to attr_model arguments
        
        eval_type: roar, selectivity
        model_type: cnn, cnnwithcbam
        attr_type: deconv, gradcam, guidedgrad, relavance, vanillagrad, inputgrad, guided_gradcam
        """
        if args.use_cuda:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                print("cuda is not available.")
                device = "cpu" 

        # path settings
        prj_path = Path(args.prj_path)
        sv_main_path = prj_path/"trained"/args.data_type
        record_main_path = prj_path/"trainlog"
        for p in [sv_main_path, record_main_path]:
            self.check_path_exist(p, directory=True)
        record_path = record_main_path/f"{args.record_file}.txt"
        self.record_result(record_path, create=True)
        
        # start
        load_path = None
        self.masks_dict = None  # need to initialize
        delete_percentages = [round(x.item(), 2) for x in torch.arange(0.1, 1, 0.1)]
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        for m_type in args.model_type:
            # select a model class
            model_class = self.model_dict[args.data_type][m_type]
            # first training
            sv_attr_path = sv_main_path/args.eval_type
            self.check_path_exist(sv_attr_path, directory=True)
            first_sv_path = str(sv_attr_path/f"{m_type}-first.pt")
            train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=True)
            model = model_class()
            model = model.to(device)
            first_best_acc = self.main_train(model, train_loader, test_loader, args.n_step, first_sv_path, device)
            
            for a_type in args.attr_type:
                # select a attribution method class
                attr_class = self.attr_dict[a_type]
                # initialize masks_dict
                self.masks_dict = self.init_masks_dict(args, train_dataset, test_dataset)           
                # kwargs to attribution model
                kwargs = self.get_kwargs_to_attr_model(args.data_type, m_type, a_type)
                # record the first trained result for the each attribution type begins
                self.record_result(record_path, create=False, model_type=m_type, del_p=0.0,
                    attr_type=a_type, best_acc=first_best_acc)
                load_path = first_sv_path
                for del_p in delete_percentages:
                    print(f"[Training {m_type}] manual_seed={args.seed}\n")
                    print(f"[Alert] Attribution type: {a_type} / Deletion of inputs: {del_p}")
                    # save path settings for each attribution, deleted percentages
                    sv_path = str(sv_attr_path/f"{m_type}-{a_type}-{del_p}.pt")
                    # start retraining
                    if args.eval_type == "roar":
                        best_acc = self.roar(args, model_class, attr_class, del_p, sv_path, load_path, device, **kwargs)
                    elif args.eval_type == "selectivity":
                        raise NotImplementedError
                        # self.selectivity(args, model_class, attr_class, del_p, device)
                    else:
                        raise NotImplementedError
                    # sv_path will be next load_path 
                    load_path = sv_path
                    # record best model accruacy automatically
                    self.record_result(record_path, create=False, model_type=m_type, del_p=del_p,
                        attr_type=a_type, best_acc=best_acc)
                # save after all training
                torch.save(self.masks_dict, sv_attr_path/f"{m_type}-{a_type}.masks")
                del self.masks_dict
