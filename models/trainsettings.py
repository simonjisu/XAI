import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchxai import XaiTrainer
from .mnist import cnn as mnistcnn

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
                "cnn": mnistcnn.CnnModel,
                "cnnwithcbam": mnistcnn.CnnModelWithCBAM
            }
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
        test_acc = 100*(correct / len(test_loader.dataset))

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
        test_dataset = self.dataset_dict[arg.data_type](
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
    
    def get_new_data(self, args, masks_dict, del_p, typ):
        datas = masks_dict[typ][del_p]
        masks = masks_dict[typ+"-mask"][del_p]
        new_datas = datas.masked_fill(masks, 0.0)
        # B, C, H, W = datas.size()
        if args.data_type.lower() == "mnist":
            # datashape: (B, H, W)
            return new_datas.squeeze()
        elif args.data_type.lower() == "cifar10":
            # datashape: (B, H, W, C)
            return new_datas.permute(0, 2, 3, 1).numpy()

    def get_argmax_indices(self, data_loader, attr_model, masks_dict, del_p, typ):
        
        def calculate_masks(outputs, del_p):
            B, C, H, W = indices.size()
            reshaped_outputs = outputs.view(B, C, -1)
            vals, _ = reshaped_outputs.sort(-1)
            # decide how many pixels to delete
            del_n_idx = torch.LongTensor([int(del_p * H * W)])  
            del_val = vals.index_select(-1, del_n_idx)
            del_mask = (reshaped_outputs <= del_val).view(B, C, H, W)
            return del_mask

        temp = []
        temp_masks = []
        for datas, targets in data_loader:
            temp.append(datas)
            outputs = attr_model.get_attribution(datas, targets).detach()
            masks = calculate_masks(outputs, del_p)
            temp_masks.append(masks)

        all_datas = torch.cat(temp, dim=0)
        all_masks = torch.cat(temp_masks, dim=0)

        masks_dict[typ][del_p] = all_datas
        previous_del_p = list(masks_dict[typ+"-masks"])[-1]
        masks_dict[typ+"-masks"][del_p] = train_masks + masks_dict[typ+"-masks"][previous_del_p]

        return masks_dict

    def create_attr_model(self, model_class, attr_class, load_path):
        """
        recreate the model & load its weights. after this create an attribution model
        """
        model = model_class()
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        attr_model = attr_class(model)
        return model, attr_model

    def evaluation(self, args, attr_model, del_p, masks_dict):
        train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=False)
        masks_dict = self.get_argmax_indices(train_loader, attr_model, masks_dict, del_p, 
                                            typ="train")
        masks_dict  = self.get_argmax_indices(test_loader, attr_model, masks_dict, del_p, 
                                            typ="test")

        train_dataset.data = self.get_new_data(args, masks_dict, del_p, typ="train")
        test_dataset.data = self.get_new_data(args, masks_dict, del_p, typ="test")
        # recreate new data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, 
            shuffle=True)
        return train_loader, test_loader, masks_dict

    def roar(self, args, model_class, attr_class, del_p, sv_attr_path, device):
        sv_path = str(sv_attr_path/f"{m_type}-{a_type}-{del_p}.pt")
        if del_p == 0.0:
            # first training
            train_dataset, test_dataset, train_loader, test_loader = self.build_dataset(args, shuffle=True)
            masks_dict = self.init_masks_dict(args, train_dataset, test_dataset)
            model = model_class()
            load_path = sv_path
        else:
            # after deletion retrain
            model, attr_model = self.create_attr_model(model_class, attr_class, load_path)
            train_loader, test_loader, masks_dict = self.evaluation(args, attr_model, del_p, masks_dict)
        # start to train model
        model = model.to(device)
        best_acc = self.main_train(model, train_loader, test_loader, args.n_step, sv_path, device)
        
        # record best model accruacy automatically
        self.record_result(self.record_path, create=False, 
            model_type=m_type, attr_method_type=a_type, best_acc=best_acc)

        torch.save(masks_dict, sv_attr_path/f"{m_type}-{a_type}.masks")

    def selectivity(self, args, model_class, attr_class, del_p, sv_attr_path, device):
        raise NotImplementedError

    def record_result(self, record_path, create=False, **kwargs):
        if create:
            self.check_path_exist(record_path, directory=False)
            with record_path.open(mode="w", encoding="utf-8") as f:
                f.write("| model_type | attr_method_type | best_acc |\n")
                f.write("|--|--|--|\n")
        else:
            model_type = kwargs["model_type"]
            attr_method_type = kwargs["attr_method_type"]
            best_acc= kwargs["best_acc"]
            with record_path.open(mode="a", encoding="utf-8") as f:
                f.write(f"|{model_type}|{attr_method_type}|{best_acc:.2f}%|\n")

    def main(self, args):
        """
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
        self.sv_main_path = prj_path/"trained"/args.data_type
        record_main_path = prj_path/"trainlog"
        for p in [sv_main_path, record_main_path]:
            self.check_path_exist(p, directory=True)
        self.record_path = record_main_path/f"{record_file}.txt"
        self.record_result(self.record_path, create=True)
        
        # start
        delete_percentages = [round(x.item(), 2) for x in torch.arange(0, 1, 0.1)]
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        for m_type in args.model_type:
            # select a model class
            model_class = self.model_dict[args.data_type][m_type]
            for a_type in args.attr_type:
                # select a attribution method class
                attr_class = self.attr_dict[a_type]                
                
                for del_p in delete_percentages:
                    print(f"[Training {m_type}] manual_seed={args.seed}\n")
                    print(f"[Alert] Attribution type: {a_type} / Deletion of inputs: {del_p}")
                    # save path settings for each attribution, evaluation methods
                    sv_attr_path = self.sv_main_path/args.eval_type
                    self.check_path_exist(sv_attr_path, directory=True)
                    # start training
                    if args.eval_type == "roar":
                        self.roar(args, model_class, attr_class, del_p, sv_attr_path, device)
                    elif args.eval_type == "selectivity":
                        self.selectivity(args, model_class, attr_class, del_p, device)
                    else:
                        raise NotImplementedError

