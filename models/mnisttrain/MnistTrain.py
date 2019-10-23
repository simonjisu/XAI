# https://arxiv.org/abs/1711.06104
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from MnistModels import MNISTmodel

def build_dataset(root, batch_size, download=False):
    train_dataset = datasets.MNIST(
        root=root,                                
        train=True,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))  # normalize to (-1, 1)
        ]),
        download=download)
    test_dataset = datasets.MNIST(
        root=root, 
        train=False,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
        ]),
        download=download)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=True)
    return train_dataset, test_dataset, train_loader, test_loader

def train(model, train_loader, optimizer, loss_function, logterm, device):
    train_loss = 0
    
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # record
        if logterm >= 1:
            if i % logterm == 0:    
                print(f"[Log] Progress: {100*i/len(train_loader):.2f}% Batch Average loss: {loss:.4f}")
                
    return train_loss

def test(model, test_loader, loss_function, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_acc = 100*(correct / len(test_loader.dataset))

    return test_loss, test_acc

def main_train(model, train_loader, test_loader, n_step, logterm, sv_path, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0
    for step in range(n_step):
        train_loss = train(model, train_loader, optimizer, loss_function, logterm, device)
        test_loss, test_acc = test(model, test_loader, loss_function, device)
        print(f"[Step] {step+1}/{n_step}")
        print(f"[Train] Average loss: {train_loss:.4f}")
        print(f"[Test] Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), sv_path)
            print("[Alert] best model saved")
        print("----"*10)
    return best_acc
    
def main(model_types, activation_types, **kwargs):
    root = kwargs["root"]
    project_path = kwargs["project_path"]
    logterm = kwargs["logterm"]
    paper_num = kwargs["paper_num"]
    sv_folder = kwargs["sv_folder"]
    n_step = kwargs["n_step"]
    batch_size = kwargs["batch_size"]
    download = kwargs["download"]
    device = kwargs["device"]
    seed = kwargs["seed"]
    
    sv_main_path = project_path/sv_folder
    if not sv_main_path.exists():
        sv_main_path.mkdir()
    record_path = project_path/"trainlog"/f"{paper_num}-record.txt"
    if not record_path.exists():
        record_path.touch()
    with record_path.open(mode="w", encoding="utf-8") as f:
        f.write("| model_type | activation_type | best_acc |\n")
        f.write("|--|--|--|\n")
    # build datasets
    *_, train_loader, test_loader = build_dataset(root, batch_size, download)
    # start
    for model_type in model_types:
        for activation_type in activation_types:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(f"[Training {model_type}-{activation_type}] manual_seed={seed}\n")
            sv_path = sv_main_path/f"{model_type}-{activation_type}.pt"
            # create model
            model = MNISTmodel(model_type, activation_type).to(device)
            # start to train model
            best_acc = main_train(model, train_loader, test_loader, n_step, logterm, str(sv_path), device)
            # record best model accruacy automatically
            with record_path.open(mode="a", encoding="utf-8") as f:
                f.write(f"|{model_type}|{activation_type}|{best_acc}%|\n")
            
if __name__ == "__main__":
    
    args = dict(
        root = str(Path().home()/"code"/"data"),
        project_path = Path().home()/"code"/"XAI",
        logterm = False, 
        paper_num = "no1",
        sv_folder = "trained/mnist", 
        n_step = 20,
        batch_size = 128,
        download = False,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        seed = 73
    )
    
    activation_types = ["relu", "tanh", "sigmoid", "softplus"]
    model_types = ["dnn", "cnn"]
    main(model_types, activation_types, **args)