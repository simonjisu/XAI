__author__ = "simonjisu"
# models/mnist

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mnistModels import MNISTmodel
from mnistTrain import build_dataset, train, test, main_train
from collections import OrderedDict
from relavance.lrp import LRP
from saliency.saliency import GradInput

def create_attr_model(attr_type, model_type, activation_type, load_path, device):
    assert attr_type in ["lrp", "gradinput"], '`attr_type` must be ["lrp", "gradinput"]'
    attr_dict = {
        "lrp" : LRP,
        "gradinput" : GradInput
    }
    model = MNISTmodel(model_type, activation_type)
    model.load_state_dict(torch.load(load_path, map_location="cpu"))
    model.eval()
    # LRP, GradInput
    attr_model = attr_dict[attr_type](model)
    return model, attr_model


def estimate(attr_model, sel_step, test_dataset, indices_dict, batch_size):
    
    def get_max_indices(datas, targets, attr_model):
        outputs = attr_model.get_attribution(datas, targets)
        B, C, H, W = outputs.size()
        argmax_v = outputs.view(B, C, -1).argmax(-1)
        row_max = argmax_v // W
        col_max = argmax_v % W
        max_indices = torch.cat([row_max, col_max], dim=-1)
        return max_indices

    def calculate_masks(masks, indices, del_idx, C, H, W):
        return (masks+indices.eq(del_idx).view(-1, C, H, W)).ge(1).byte()
    
    if train_dataset.data.ndimension() == 3:
        train_datas = train_dataset.data.float().unsqueeze(1)
        train_targets = train_dataset.targets
        test_datas = test_dataset.data.float().unsqueeze(1)
        test_targets = test_dataset.targets
    _, C, H, W = train_datas.size()

    # del_idx: decide how many pixels to delete
    del_idx = int(del_p * H * W)
    
    max_indices = get_argmax_indices(test_datas, train_targets, attr_model)
    test_indices = get_argmax_indices(test_datas, test_targets, attr_model)
    
    # rebuild dataset
    masks_dict["train"][del_p] = calculate_masks(masks_dict["train"][del_p], train_indices, del_idx, C, H, W)
    train_dataset.data = (train_datas * masks_dict["train"][del_p].eq(0)).squeeze()
    
    masks_dict["test"][del_p] = calculate_masks(masks_dict["test"][del_p], test_indices, del_idx, C, H, W)
    test_dataset.data = (test_datas * masks_dict["test"][del_p].eq(0)).squeeze()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=True)
    
    return train_dataset, test_dataset, train_loader, test_loader


def main(model_types, activation_types, attr_types, **kwargs):
    root = kwargs["root"]
    project_path = kwargs["project_path"]
    logterm = kwargs["logterm"]
    record_name = kwargs["record_name"]
    sv_folder = kwargs["sv_folder"]
    n_step = kwargs["n_step"]
    batch_size = kwargs["batch_size"]
    download = kwargs["download"]
    device = kwargs["device"]
    seed = kwargs["seed"]
    # selectivity
    select_n_step = kwargs["select_n_step"]

    sv_main_path = project_path/sv_folder
    if not sv_main_path.exists():
        sv_main_path.mkdir()
    record_path = project_path/"trainlog"/f"{record_name}-record.txt"
    if not record_path.exists():
        record_path.touch()
    with record_path.open(mode="w", encoding="utf-8") as f:
        f.write("| model_type | activation_type | attr_type | select_step | best_acc |\n")
        f.write("|--|--|--|--|--|\n")
    
    # mnist dataset delete idx
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # start
    for model_type in model_types:
        for activation_type in activation_types:
            print(f"[Alert] Training {model_type}-{activation_type} manual_seed={seed}")
            for attr_type in attr_types:
                # build datasets
                _, test_dataset, _, test_loader = build_dataset(root, batch_size, download)
                # setting load_path
                load_path = sv_main_path/f"{model_type}-{activation_type}.pt"
                # create model(loaded), attr_model
                model, attr_model = create_attr_model(attr_type, model_type, activation_type, load_path, device)
                indices_dict = OrderedDict()
                # start training from select_n_step
                for sel_step in range(select_n_step):

                    print(f"[Alert] Attribution type: {attr_type} / Selection Step: {sel_step+1}")
                    indices_dict[sel_step]
                    # delete below
                    if del_p != 0.0:
                        masks_dict["train"][del_p] = masks_dict["train"][prev_del_p]
                        masks_dict["test"][del_p] = masks_dict["test"][prev_del_p]
                        
                        # create attr_model
                        model, attr_model = create_attr_model(attr_type, model_type, activation_type, load_path, device)
                        
                        # estimate and rebuild dataset
                        train_dataset, test_dataset, train_loader, test_loader = estimate(
                            attr_model, del_p, train_dataset, test_dataset, masks_dict, batch_size)
                    else:
                        # create new model at first time
                        model = MNISTmodel(model_type, activation_type)
                        masks_dict["train"][del_p] = torch.zeros_like(train_dataset.data.unsqueeze(1))
                        masks_dict["test"][del_p] = torch.zeros_like(test_dataset.data.unsqueeze(1))

                    # start to train model
                    model = model.to(device)
                    best_acc = main_train(model, train_loader, test_loader, n_step, logterm, str(sv_path), device)
                    load_path = sv_path
                    prev_del_p = del_p
                    # record best model accruacy automatically
                    with record_path.open(mode="a", encoding="utf-8") as f:
                        f.write(f"|{model_type}|{activation_type}|{attr_type}|{del_p}|{best_acc:.2f}%|\n")
                    
                torch.save(masks_dict["test"], sv_attr_path/f"{model_type}-{activation_type}.masks")
            
if __name__ == "__main__":
    
    args = dict(
        root = str(Path().home()/"code"/"data"),
        project_path = Path().home()/"code"/"XAI",
        logterm = False, 
        record_name = "mnist-selectivity",
        sv_folder = "trained/mnist", 
        n_step = 10,
        batch_size = 512,
        download = False,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        seed = 73, 
        select_n_step = 50
    )
    
#     activation_types = ["relu", "tanh", "sigmoid", "softplus"]
#     model_types = ["dnn", "cnn"]
    activation_types = ["relu"]
    model_types = ["cnn"]
    attr_types = ["lrp", "gradinput"]
    main(model_types, activation_types, attr_types, **args)