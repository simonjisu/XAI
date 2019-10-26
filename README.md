# XAI

* eXplainable AI
* AI college Recording Repo

# Requirements

```
pytorch >= 1.3.0
torchvision == 0.4.1
```

## Paper Implementation

### [No.1] Towards better understanding of gradient-based attribution methods for Deep Neural Networks

* Link: [https://arxiv.org/abs/1711.06104](https://arxiv.org/abs/1711.06104)
    
to start train model run code:
> ```bash
> python -u ./models/mnist/MnistTrain.py
> ```
    
see training log at `./trainlog/no1.log`
    
| model_type | activation_type | best_acc |
|--|--|--|
|dnn|relu|98.33%|
|dnn|tanh|97.65%|
|dnn|sigmoid|98.08%|
|dnn|softplus|98.20%|
|cnn|relu|99.16%|
|cnn|tanh|98.57%|
|cnn|sigmoid|98.88%|
|cnn|softplus|98.9%|

### TODO

- [x] Implement 'DeconvNet'
- [x] Implement 'LRP'
- [x] Implement 'Vanilla Saliency'
- [ ] Implement 'Integrated Gradients'
- [ ] Implement 'DeepLIFT'
- [ ] Implement 'evaluation metric'
- [ ] Build a Visualization Demo