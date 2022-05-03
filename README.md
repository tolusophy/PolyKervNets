# PolyKervNets
PolyKervNets: Activation-free Neural Networks For Efficient Private Inference

In this work, we propose a FHE and MPC suitable/efficient DNN architecture based on optimized polynomial kervolution from Kervolutional Neural Network (https://github.com/wang-chen/kervolution/) called _**PolyKervNet**_, which completely eliminates the need for non-linear activation and max pooling layers.

## How to use
1. Clone this repository

```
git clone https://github.com/Ti-Oluwanimi/PolyKervNets.git
```

2. Pip/Conda install the following packages
    - Pytorch (v1.10, v1.11)
    - TenSEAL
    - CryptEN
    
4. Edit main_cifar.py and run.sh

In the ```main.py```, you can specify the network you want to train(for example):

```
model = alexkerv(num_classes=10)

##Note
Please contact me if there are issues within the codebase. 
