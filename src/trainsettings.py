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
        self.default_kwargs = dict(norm_mode=1)
        # self.default_kwargs = {"<none>": None}
        self.kwargs_packs = {
            "mnist": {
                "gradcam": {
                    "cnn": dict(layers_name=None, norm_mode=1),
                    "resnet": dict(layers_name="relu_last", norm_mode=1),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=1),
                    "resnetanr": dict(layers_name="relu_last", norm_mode=1)
                },
                "guidedgrad": {
                    "cnn": dict(act=nn.ReLU, norm_mode=1),
                    "resnet": dict(act=nn.ReLU, norm_mode=1),
                    "resnetcbam": dict(act=nn.ReLU, norm_mode=1),
                    "resnetanr": dict(act=nn.ReLU, norm_mode=1)
                },
                "relavance": {
                    "cnn": dict(use_rho=False, norm_mode=1)
                },
                "deconv": {
                    "cnn": dict(module_name="convs", norm_mode=1)
                }, 
                # if None > will be set to default_kwargs
                "vanillagrad": None,
                "inputgrad": None,
                "random": None
            },
            "cifar10": {
                "gradcam": {
                    "resnet": dict(layers_name="relu_last", norm_mode=1),
                    "resnetcbam": dict(layers_name="relu_last", norm_mode=1),
                    "resnetanr": dict(layers_name="relu_last", norm_mode=1)
                },
                "guidedgrad": {
                    "resnet": dict(act=nn.ReLU, norm_mode=1),
                    "resnetcbam": dict(act=nn.ReLU, norm_mode=1),
                    "resnetanr": dict(act=nn.ReLU, norm_mode=1)
                },
                "vanillagrad": None,
                "inputgrad": None,
                "random": None
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
    
    def calculate_masks(self, outputs, del_p):
        """
        importance ranking masks by percentages, always descending option!!!
        
        outputs = attributions
        return (B, C, H, W) bool tensor
        """
        B, C, H, W = outputs.size()
        reshaped_outputs = outputs.view(B, -1)
        # outputs, C = self.convert_scale(outputs, C)
        # reshaped_outputs = outputs.view(B, C, -1)
        # del_n_idx = torch.LongTensor([int(del_p * H * W)])
        # delete_idxes = idxes[:, :, :del_n_idx]
        # masks = torch.zeros((B, C, H*W), dtype=torch.bool).scatter(-1, delete_idxes, True)

        # some reference: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/data_input.py#L146
        # but not activate in this code
        # reshaped_outputs += 1e-6  # add small epsilon
        
        idxes = reshaped_outputs.argsort(-1, descending=True)
        del_n_idx = torch.LongTensor([int(del_p * C * H * W)])  # Mnist 28*28 / Cifar10 3*32*32
        delete_idxes = idxes[:, :del_n_idx]
        masks = torch.zeros((B, C*H*W), dtype=torch.bool).scatter(-1, delete_idxes, True)
        return masks.view(B, C, H, W)

    def convert_scale(self, outputs):
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
            return outputs
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
                outputs = weighted_sum(outputs.float(), w)
                if len(outputs.size()) == 3:
                    outputs = outputs.unsqueeze(1)
                
                return outputs.byte()
            else:
                return outputs

    def get_masks(self, data_loader, attr_model, percentages, typ):
        """
        Get masks to delete and corresponding datas
        
        """
        if self.data_type.lower() == "mnist":
            tf = transforms.Compose([
                    transforms.ToPILImage()
            ])
            # mnist (B, H, W) > (B, 1, H, W)
            inv_transform = lambda tensor: torch.ByteTensor([np.array(tf(x)) for x in tensor]).unsqueeze(1)
        elif self.data_type.lower() == "cifar10":
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
            if self.a_type == "random":  # Baseline 
                outputs = attr_model.get_attribution(datas, self.seed).detach()
            else:
                outputs = attr_model.get_attribution(datas, targets).detach() 
            temp_outputs.append(outputs)
        
        if self.first_eval:
            all_datas = torch.cat(temp, dim=0)
            torch.save(all_datas, str(self.sv_masks_datas / f"{self.data_type}-{typ}.data"))
        all_attributions = torch.cat(temp_outputs, dim=0)
        # if reduce dim is on, will convert to another scale (B, 1, H, W)
        all_attributions = self.convert_scale(all_attributions)

        for del_p in tqdm(percentages, 
                          desc=f"- [{typ}] masking by attributions", 
                          total=len(percentages)):
            all_masks = self.calculate_masks(all_attributions, del_p)
            # importance ranking masks by percentages, always descending option!!!
            torch.save(all_masks, str(self.sv_masks_datas / f"{self.m_type}-{self.a_type}-{typ}-{del_p}.masks"))

    def fill_datas_by_masks(self, datas, masks):
        """
        datas: (B, C, H, W) ByteTensor
        masks: (B, C, H, W) BoolTensor

        returns: (B, C, H, W) ByteTensor
        """
        B, C, *_ = datas.size()
        if self.fill_global_mean:
            # global_mean_by_channel 
            global_means = datas.float().view(B, C, -1).mean(-1, keepdim=True).mean(0, keepdim=True).byte()  # (1, 3, 1)
            fill_value = global_means.unsqueeze(-1).expand(datas.size())
            mask_fn = torch.masked_scatter
        else:
            fill_value = 0
            mask_fn = torch.masked_fill
        # ROAR:
        if self.eval_type == "roar":
            new_datas = mask_fn(datas, masks, fill_value)
        elif self.eval_type == "kar":
            new_datas = mask_fn(datas, torch.eq(masks, False), fill_value)
        return new_datas


    def get_new_data(self, del_p, typ):
        """
        returns masked data
        del_p: delete percentages
        typ: whether is train or test
        """
        datas = torch.load(str(self.sv_masks_datas / f"{self.data_type}-{typ}.data")) # (B, C, H, W) ByteTensor
        masks = torch.load(str(self.sv_masks_datas / f"{self.m_type}-{self.a_type}-{typ}-{del_p}.masks"))  # (B, C, H, W) BoolTensor
        new_datas = self.fill_datas_by_masks(datas, masks)
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
        return attr_model

    def recreate_data_loader(self, args, attr_model, del_p):
        train_dataset, test_dataset, *_ = self.build_dataset(args, shuffle=False)
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

    def evaluation(self, args, model_class, attr_class, del_p, sv_path, load_path, device, attr_kwargs):
        # after deletion retrain
        attr_model = self.create_attr_model(model_class, attr_class, load_path, attr_kwargs)
        train_loader, test_loader = self.recreate_data_loader(args, attr_model, del_p)
        # start to retrain model
        print("[Alert] Start Retraining")
        # 2020.01.20 10:00 don't use transferd model when retrain, this may cause your test accuracy going up instead of going down.
        # 2020.01.20 11:00 no... it still goes up...
        model = model_class()  
        model = model.to(device)
        best_acc = self.main_train(model, train_loader, test_loader, args.n_step, sv_path, device)
        
        return best_acc

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
            kwargs = self.default_kwargs
        return kwargs

    def first_train(self, args, device):
        best_acc_dict = OrderedDict()
        for m_type in self.model_type:
            self.m_type = m_type
            print(f"[Training {m_type}] manual_seed={self.seed}\n")
            # select a model class
            model_class = self.model_dict[args.data_type][m_type]
            # first training
            sv_path = str(self.sv_main_path /f"{m_type}-first.pt")
            train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=True)
            model = model_class()
            model = model.to(device)
            best_acc = self.main_train(model, train_loader, test_loader, args.n_step, sv_path, device)
            best_acc_dict[m_type] = best_acc
        torch.save(best_acc_dict, str(self.sv_main_path / "best_acc-first.dict"))

    def second_masking(self, args, device):
        percentages = [round(x.item(), 2) for x in torch.arange(0.1, 1, 0.1)]
        print(f"[Alert] Creating Masks by deletion percentages")
        for m_type in self.model_type:
            self.m_type = m_type
            model_class = self.model_dict[self.data_type][m_type]
            for i, a_type in enumerate(self.attr_type):
                self.first_eval = not bool(i)
                self.a_type = a_type
                attr_class = self.attr_dict[a_type]
                attr_kwargs = self.get_kwargs_to_attr_model(self.data_type, m_type, a_type)
                load_path = str(self.sv_main_path /f"{m_type}-first.pt")
                attr_model = self.create_attr_model(model_class, attr_class, load_path, attr_kwargs)
                print(f"[Alert] masking {m_type} {a_type}")
                *_, train_loader, test_loader = self.build_dataset(args, shuffle=False, batch_size=512)
                self.get_masks(train_loader, attr_model, percentages, typ="train")
                self.get_masks(test_loader, attr_model, percentages, typ="test")

    def third_evaluation(self, args, device):
        best_acc_dict = torch.load(self.sv_main_path / "best_acc-first.dict")
        percentages = [round(x.item(), 2) for x in torch.arange(0.1, 1, 0.1)]         
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
                load_path = str(self.sv_main_path /f"{m_type}-first.pt")

                for del_p in percentages:
                    print(f"[Alert] Training: {m_type}")
                    print(f"[Alert] Attribution type: {a_type} / Deletion of inputs: {del_p}")
                    # save path settings for each attribution, deleted percentages
                    p_text = f"{m_type}-{a_type}-{del_p}"
                    if self.fill_global_mean:
                        p_text += "-fgm.pt"
                    else:
                        p_text += ".pt"
                    sv_path = str(self.sv_attr_path / p_text)
                    # start retraining
                    best_acc = self.evaluation(args, model_class, attr_class, del_p, sv_path, load_path, device, attr_kwargs)
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
        self.fill_global_mean = args.fill_global_mean
        # path settings
        self.prj_path = Path(args.prj_path)
        
        self.sv_main_path = self.prj_path/"trained"/self.data_type
        self.sv_masks_datas = self.sv_main_path / "masksdatas"
        self.record_main_path = self.prj_path/"trainlog"
        # save all models.pt by deleteion/recover/selection : sv_main_path > eval_type > rcd
        if self.reduce_color_dim is not None:
            self.sv_attr_path = self.sv_main_path/self.eval_type/self.reduce_color_dim
        else:
            self.sv_attr_path = self.sv_main_path/self.eval_type/"plain"
        for p in [self.sv_main_path, self.sv_masks_datas, self.record_main_path, self.record_main_path, self.sv_attr_path]:
            self.check_path_exist(p, directory=True)
        self.record_path = self.record_main_path/f"{args.record_file}.txt"
    
        # start        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if not args.skip_first_train:
            self.first_train(args, device)
        if not args.skip_second_masking:
            self.second_masking(args, device)
        if not args.skip_third_eval:
            self.record_result(self.record_path, create=True)
            self.third_evaluation(args, device)