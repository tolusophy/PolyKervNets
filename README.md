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
   ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```
    
    - TenSEAL
    ```pip install tenseal```
    
    - CryptEN
    ```pip install crypten```
    
3. Open the MNIST ipynb file to see an example of using TenSEAL CKKS library to encrypt our Poly1Net and infer from MNIST. To learn how to use TenSEAL to encrypt your custom architectures and data, check the tutorials at https://github.com/OpenMined/TenSEAL
4. You can also

In the ```main.py```, you can specify the network you want to train(for example):

```
model = alexkerv(num_classes=10)

##Note
Please contact me if there are issues within the codebase. 
