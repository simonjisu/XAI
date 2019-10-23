import torch
import torch.nn as nn

class deconvMNIST(nn.Module):
    def __init__(self, model):
        super(deconvMNIST, self).__init__()
                
        # deconv
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type
        self.convs_len = model.convs_len
        self.fc_len = model.fc_len
        if self.model_type == "cnn":
            self.switches = model.switches
        self.convs, self.fc = self.deconv_make_layers()
        
    def deconv_make_layers(self):
        if self.model_type.lower() == "dnn":
            convs = None
            fc = nn.Sequential(
                nn.Linear(10, 512),
                self.activation_func(), 
                nn.Linear(512, 512),
                self.activation_func(), 
                nn.Linear(512, 28*28),
            )
        elif self.model_type.lower() == "cnn":
            fc = nn.Sequential(
                nn.Linear(10, 128),
                self.activation_func(),
                nn.Linear(128, 64*12*12)    
            )
            convs = nn.Sequential(
                nn.MaxUnpool2d(2),
                self.activation_func(), 
                nn.ConvTranspose2d(64, 32, 3),
                self.activation_func(),
                nn.ConvTranspose2d(32, 1, 3),
            )
        else:
            assert False, "[init error] `model` doesn't have `model_type` or `activation_func` attrs"
        return convs, fc
    
    def forward(self, t):
        x = self.fc(t)
        if self.convs is not None:
            x = x.view(x.size(0), 64, 12, 12)
            for idx, layer in enumerate(self.convs):
                if isinstance(layer, nn.MaxUnpool2d):
                    x = layer(x, self.switches[self.convs_len - 1 - idx])
                else:
                    x = layer(x)
        return x
    
    # TODO: write code for a specific layer's activation map