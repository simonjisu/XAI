from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchxai.trainer import XaiTrainer
from .models import plaincnn, resnet
from collections import OrderedDict
from torchxai.module.anr import GlobalAttentionGate
import numpy as np
import time

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
                "cnn": plaincnn.CnnMnist,
                "resnet": resnet.ResNetMnist,
                "resnetcbam": resnet.ResNetMnistCBAM,
                "resnetanr": resnet.ResNetMnistANR
            },
            "cifar10": {
                "cnn": None,
                "resnet": resnet.ResNetCifar10, 
                "resnetcbam": resnet.ResNetCifar10CBAM,
                "resnetanr": resnet.ResNetCifar10ANR
            }
        }

        # give some arguments to attribution models
        self.kwargs_packs = {
            "mnist": {
                "gradcam": {
                    "cnn": dict(layers_name=None, norm_mode=1),
                    "resnet": dict(layers_name="relu_last", norm_mode=1),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=1),
                    "resnetanr": dict(layers_name="relu_last", norm_mode=1)
                },
                "guidedgrad": {
                    "cnn": dict(act=nn.ReLU),
                    "resnet": dict(act=nn.ReLU),
                    "resnetcbam": dict(act=nn.ReLU),
                    "resnetanr": dict(act=nn.ReLU)
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
                    "resnet": dict(layers_name="relu_last", norm_mode=1),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=1),
                    "resnetanr": dict(layers_name="relu_last", norm_mode=1)
                },
                "guidedgrad": {
                    "resnet": dict(act=nn.ReLU),
                    "resnetcbam": dict(act=nn.ReLU),
                    "resnetanr": dict(act=nn.ReLU)
                },
                "vanillagrad": None, 
                "inputgrad": None
            }
            
        }

        self.loss_fn_dict = {
            "anr": GlobalAttentionGate.loss_function,
            "crossentropy": F.cross_entropy
        }

    def _cal_time(self, s, e):
        total_time = e-s
        hour = int(total_time // (60*60))
        minute = int((total_time - hour*60*60) // 60)
        second = total_time - hour*60*60 - minute*60
        txt = f"[Alert] Training Excution time with validation: {hour:d} h {minute:d} m {second:.4f} s"
        print(txt)

    def _cal_loss(self, model, loss_function, datas, targets, reduction="mean"):
        outputs = model(datas)
        if "anr" in self.m_type:
            loss = loss_function(outputs, targets, model.reg_loss, reduction=reduction)
        else:
            loss = loss_function(outputs, targets, reduction=reduction)
        if model.training:
            return loss
        else:
            return loss, outputs

    def train(self, model, train_loader, optimizer, loss_function, device):
        train_loss = 0
        model.train()
        for datas, targets in train_loader:
            datas, targets = datas.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = self._cal_loss(model, loss_function, datas, targets, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
                    
        return train_loss

    def test(self, model, test_loader, loss_function, device):
        model.eval()
        test_loss = 0
        corrects = 0
        with torch.no_grad():
            for datas, targets in test_loader:
                datas, targets = datas.to(device), targets.to(device)
                loss, outputs = self._cal_loss(model, loss_function, datas, targets, reduction="sum")
                test_loss += loss.item()
                preds = outputs.argmax(dim=1, keepdim=True)
                corrects += preds.eq(targets.view_as(preds)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_acc = 100*(corrects / len(test_loader.dataset))

        return test_loss, test_acc

    def main_train(self, model, train_loader, test_loader, n_step, sv_path, device):
        if "anr" in self.m_type:
            loss_function = self.loss_fn_dict["anr"]
        else:
            loss_function = self.loss_fn_dict["crossentropy"]
        
        if self.data_type == "mnist":
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            # self.scheduler = None
        elif self.data_type == "cifar10":
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CyclicLR
            # self.scheduler = optim.lr_scheduler.CyclicLR(optimizer, mode="exp_range",
            #    base_lr=0.001, max_lr=0.01, step_size_up=2000, cycle_momentum=False)
        best_acc = 0.0
        start_time = time.time()
        for step in range(n_step):
            train_loss = self.train(model, train_loader, optimizer, loss_function, device)
            test_loss, test_acc = self.test(model, test_loader, loss_function, device)
            print(f"[Step] {step+1}/{n_step}")
            print(f"[Train] Average loss: {train_loss:.4f}")
            print(f"[Test] Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), sv_path)
                print("[Alert] best model saved")
            print("----"*10)
        end_time = time.time()
        self._cal_time(start_time, end_time)
        return best_acc

    def build_dataset(self, args, shuffle=True, batch_size=None):
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
        batch_size = args.batch_size if batch_size is None else batch_size
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=shuffle)
        return train_dataset, test_dataset, train_loader, test_loader

    def check_if_model_supported(self, args):
        # relavance method is restricted to use custom models
        pass
    
    def convert_scale(self, outputs, C):
        """
        reference: https://en.wikipedia.org/wiki/Grayscale
        methods: mean, rec601, itu_r_bt707, itu_r_bt2100
        [0: mean] mean
        mean channels
        [1: rec601] rec601
        In the Y'UV and Y'IQ models used by PAL and NTSC, 
        the rec601 luma component is computed as
        Y = 0.299*R + 0.587*G + 0.114*B
        [2: itu_r_bt707] itu-r_bt.707
        The ITU-R BT.709 standard used for HDTV developed by the ATSC 
        uses different color coefficients, computing the luma component as
        Y = 0.2126*R + 0.7152*G + 0.0722*B
        [3: itu_r_bt2100] itu-r_bt.2100
        The ITU-R BT.2100 standard for HDR television uses yet different 
        coefficients, computing the luma component as
        Y = 0.2627*R + 0.6780*G + 0.0593*B
        """
        if outputs.size(1) == 1:
            return outputs, C        
        else:
            method_dict = {
                "mean": [1/3, 1/3, 1/3],
                "rec601": [0.299, 0.587, 0.114],
                "itu_r_bt707": [0.2126, 0.7152, 0.0722],
                "itu_r_bt2100": [0.2627, 0.6780, 0.0593]
            }
            def weighted_sum(x, w):
                w = torch.FloatTensor(w).unsqueeze(0).repeat(x.size(0), 1).unsqueeze(-1).unsqueeze(-1)
                return (x * w).sum(1)
            if self.reduce_color_dim is not None:
                w = method_dict[self.reduce_color_dim]
                outputs = weighted_sum(outputs, w)
                if len(outputs.size()) == 3:
                    outputs = outputs.unsqueeze(1)
                C = 1
                return outputs, C
            else:
                return outputs, C

    def calculate_masks(self, outputs, del_p):
        B, C, H, W = outputs.size()
        outputs, C = self.convert_scale(outputs, C)
        reshaped_outputs = outputs.view(B, C, -1)
        vals, _ = reshaped_outputs.sort(-1, descending=True)
        # decide how many pixels to delete
        del_n_idx = torch.LongTensor([int(del_p * H * W)])  
        del_vals = vals.index_select(-1, del_n_idx)
        del_masks = (reshaped_outputs >= del_vals).view(B, C, H, W)
        return del_masks

    def get_masks(self, args, data_loader, attr_model, delete_percentages, typ):
        """
        Get masks to delete and corresponding datas
        """
        
        if args.data_type.lower() == "mnist":
            tf = transforms.Compose([
                    transforms.ToPILImage()
            ])
            # mnist (B, H, W) > (B, 1, H, W)
            inv_transform = lambda tensor: torch.ByteTensor([np.array(tf(x)) for x in tensor]).unsqueeze(1)
        elif args.data_type.lower() == "cifar10":
            inv_normalize = transforms.Normalize(
                    mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                    std=[1/0.2470, 1/0.2435, 1/0.2616]
                )
            tf = transforms.Compose([
                inv_normalize, 
                transforms.ToPILImage()
            ])
            # cifar10 (B, H, W, C) > (B, C, H, W)
            inv_transform = lambda tensor: torch.ByteTensor([np.array(tf(x)) for x in tensor]).permute(0, 3, 1, 2)
        temp = []
        temp_outputs = []
        temp_masks = []
        for datas, targets in tqdm(data_loader, 
                                   desc=f"- [{typ}] calulating attributions", 
                                   total=len(data_loader)):
            if self.first_eval:
                origin_datas = inv_transform(datas)
                temp.append(origin_datas)
            outputs = attr_model.get_attribution(datas, targets).detach()
            temp_outputs.append(outputs)
        
        if self.first_eval:
            all_datas = torch.cat(temp, dim=0)
            torch.save(all_datas, str(self.sv_attr_path / f"{self.data_type}-{typ}.data"))
        all_attributions = torch.cat(temp_outputs, dim=0)    

        for del_p in tqdm(delete_percentages, 
                          desc=f"- [{typ}] deleting by attributions", 
                          total=len(delete_percentages)):
            all_masks = self.calculate_masks(all_attributions, del_p)
            torch.save(all_masks, str(self.sv_attr_path / f"{self.m_type}-{self.a_type}-{typ}-{del_p}.masks"))

    def get_new_data(self, del_p, typ):
        """
        returns masked data
        del_p: delete percentages
        typ: whether is train or test

        if `self.reduce_color_dim` option is not None:
            recude all channel dimention to 1
        """
        datas = torch.load(str(self.sv_attr_path / f"{self.data_type}-{typ}.data"))
        masks = torch.load(str(self.sv_attr_path / f"{self.m_type}-{self.a_type}-{typ}-{del_p}.masks"))
        new_datas = datas.masked_fill(masks, 0.0)
        # B, C, H, W = datas.size()
        if self.data_type.lower() == "mnist":
            # datatype: torch.unit8, datashape: (B, H, W)
            return new_datas.squeeze(1)
        elif self.data_type.lower() == "cifar10":
            # datatype: numpy.unit8, datashape: (B, H, W, C)
            return new_datas.permute(0, 2, 3, 1).numpy()

    def create_attr_model(self, model_class, attr_class, load_path, attr_kwargs):
        """
        recreate the model & load its weights. after this create an attribution model
        """
        model = model_class()
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        attr_model = attr_class(model, **attr_kwargs)
        return model, attr_model

    def evaluation(self, args, attr_model, del_p):
        train_dataset, test_dataset, *_ = self.build_dataset(args, shuffle=False)
        # self.get_masks(args, train_loader, attr_model, del_p, typ="train")
        # self.get_masks(args, test_loader, attr_model, del_p, typ="test")

        train_dataset.data = self.get_new_data(del_p, typ="train")
        test_dataset.data = self.get_new_data(del_p, typ="test")
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

    def roar(self, args, model_class, attr_class, del_p, sv_path, load_path, device, attr_kwargs):
        # after deletion retrain
        model, attr_model = self.create_attr_model(model_class, attr_class, load_path, attr_kwargs)
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

    def first_train(self, args, device):
        best_acc_dict = OrderedDict()
        
        for m_type in self.model_type:
            self.m_type = m_type
            print(f"[Training {m_type}] manual_seed={self.seed}\n")
            # select a model class
            model_class = self.model_dict[args.data_type][m_type]
            # first training
            sv_path = str(self.sv_attr_path/f"{m_type}-first.pt")
            train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=True)
            model = model_class()
            model = model.to(device)
            best_acc = self.main_train(model, train_loader, test_loader, args.n_step, sv_path, device)
            best_acc_dict[m_type] = best_acc
        torch.save(best_acc_dict, str(self.sv_attr_path / "best_acc-first.dict"))

    def second_evaluation(self, args, device):
        best_acc_dict = torch.load(self.sv_attr_path / "best_acc-first.dict")
        delete_percentages = [round(x.item(), 2) for x in torch.arange(0.1, 1, 0.1)]
        
        print(f"[Alert] Creating Masks by deletion percentages")
        for m_type in self.model_type:
            self.m_type = m_type
            model_class = self.model_dict[self.data_type][m_type]
            for i, a_type in enumerate(self.attr_type):
                self.first_eval = not bool(i)
                self.a_type = a_type
                attr_class = self.attr_dict[a_type]
                attr_kwargs = self.get_kwargs_to_attr_model(self.data_type, m_type, a_type)
                load_path = str(self.sv_attr_path/f"{m_type}-first.pt")
                _, attr_model = self.create_attr_model(model_class, attr_class, load_path, attr_kwargs)
                print(f"[Alert] {m_type} {a_type}")
                *_, train_loader, test_loader = self.build_dataset(args, shuffle=False, batch_size=512)
                self.get_masks(args, train_loader, attr_model, delete_percentages, typ="train")
                self.get_masks(args, test_loader, attr_model, delete_percentages, typ="test")
                

        print(f"[Alert] Retraining by deletion percentages")
        for m_type in self.model_type:
            self.m_type = m_type
            model_class = self.model_dict[self.data_type][m_type]  # select a model class
            first_best_acc = best_acc_dict[m_type]
            for a_type in self.attr_type:
                self.a_type = a_type
                attr_class = self.attr_dict[a_type]  # select a attribution method class
                attr_kwargs = self.get_kwargs_to_attr_model(self.data_type, m_type, a_type)  # kwargs to attribution model
                # record the first trained result for the each attribution type begins
                self.record_result(self.record_path, create=False, model_type=m_type, del_p=0.0,
                    attr_type=a_type, best_acc=first_best_acc)
                load_path = str(self.sv_attr_path/f"{m_type}-first.pt")

                for del_p in delete_percentages:
                    print(f"[Alert] Training: {m_type}")
                    print(f"[Alert] Attribution type: {a_type} / Deletion of inputs: {del_p}")
                    # save path settings for each attribution, deleted percentages
                    sv_path = str(self.sv_attr_path/f"{m_type}-{a_type}-{del_p}.pt")
                    # start retraining
                    if self.eval_type == "roar":
                        best_acc = self.roar(args, model_class, attr_class, del_p, sv_path, load_path, device, attr_kwargs)
                    elif self.eval_type == "selectivity":
                        raise NotImplementedError
                        # self.selectivity(args, model_class, attr_class, del_p, device)
                    else:
                        raise NotImplementedError
                    # record best model accruacy automatically
                    self.record_result(self.record_path, create=False, model_type=m_type, del_p=del_p,
                        attr_type=a_type, best_acc=best_acc)
                    
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
        self.model_type = args.model_type
        self.data_type = args.data_type
        self.eval_type = args.eval_type
        self.attr_type = args.attr_type
        self.seed = args.seed
        self.reduce_color_dim = args.reduce_color_dim
        # path settings
        self.prj_path = Path(args.prj_path)
        
        self.sv_main_path = self.prj_path/"trained"/self.data_type
        self.record_main_path = self.prj_path/"trainlog"
        if self.reduce_color_dim is not None:
            self.sv_attr_path = self.sv_main_path/self.eval_type/self.reduce_color_dim
        else:
            self.sv_attr_path = self.sv_main_path/self.eval_type
        for p in [self.sv_main_path, self.record_main_path, self.record_main_path, self.sv_attr_path]:
            self.check_path_exist(p, directory=True)
        self.record_path = self.record_main_path/f"{args.record_file}.txt"
    
        # start        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if not args.skip_first_train:
            self.first_train(args, device)
        if not args.skip_second_eval:
            self.record_result(self.record_path, create=True)
            self.second_evaluation(args, device)