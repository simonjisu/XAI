import torch
import torch.nn as nn

class MNISTmodel(nn.Module):
    """
    paper implementation 'https://arxiv.org/abs/1711.06104'
    
    model_type: 
        - DNN: 
            Linear: 28*28, 512 
            Linear: 512, 512
            Linear: 512, 10
        - CNN: 
            Conv2d: (1, 32, 3)
            Conv2d: (32, 64, 3)
            MaxPool2d: (2)
            Linear: (64*12*12, 128)
            Linear: (128, 10)

            ** Conv2d = (in_kernels, out_kernels, kernel_size)
            ** MAxPool2d = (kernel_size)

    activation_type:
        - ReLU, Tanh, Sigmoid, Softplus
    """
    def __init__(self, model_type, activation_type):
        """
        model_type: "dnn", "cnn"
        activation_type: "relu", "tanh", "sigmoid", "softplus"
        """
        super(MNISTmodel, self).__init__()
        self.act = {"relu": nn.ReLU, 
                    "tanh": nn.Tanh, 
                    "sigmoid": nn.Sigmoid, 
                    "softplus": nn.Softplus}
        
        self.convs, self.fc = self.make_layers(model_type, activation_type)
        
    def make_layers(self, model_type, activation_type):
        activation_func = self.act[activation_type.lower()]
        if model_type.lower() == "dnn":
            convs = None
            fc = nn.Sequential(
                nn.Linear(28*28, 512),
                activation_func(), 
                nn.Linear(512, 512),
                activation_func(), 
                nn.Linear(512, 10),
            )
        elif model_type.lower() == "cnn":
            convs = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                activation_func(),
                nn.Conv2d(32, 64, 3),
                activation_func(), 
                nn.MaxPool2d(2),
            )
            fc = nn.Sequential(
                nn.Linear(64*12*12, 128),
                activation_func(),
                nn.Linear(128, 10)
            )
        else:
            assert False, "please insert `model_type` = `dnn` or `cnn`"
        return convs, fc
        
    def forward(self, x):
        if self.convs is not None:
            x = self.convs(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output