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
    
3. Open the MNIST ipynb file to see an example of using TenSEAL CKKS library to encrypt our Poly1Net and infer from MNIST. To learn how to use TenSEAL to encrypt your custom architectures and data, check the tutorials [here](https://github.com/OpenMined/TenSEAL)
4. You can also use the mpc_launcher.py to run the SMPC on your model architecture. For your own architectures, edit the ```MPC.py``` file and save. Then, run ```python mpc_launcher.py``` from the terminal. To learn more about Crypten, how to use, check out the tutorials and examples [here](https://github.com/facebookresearch/CrypTen)

##Note
Please contact me if there are issues within the codebase. 
