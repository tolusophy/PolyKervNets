# PolyKervNets: Activation-free Neural Networks For Efficient Private Inference

<p align="justify"> With the advent of cloud computing, machine learning as a service (MLaaS) has become a growing phenomenon with the potential to address many real-world problems. In an untrusted cloud environment, privacy concerns of users is a major impediment to the adoption of MLaaS. To alleviate these privacy issues and preserve data confidentiality, several private inference (PI) protocols have been proposed in recent years based on cryptographic tools like Fully Homomorphic Encryption (FHE) and Secure Multiparty Computation (MPC). Deep neural networks (DNN) have been the architecture of choice in most MLaaS deployments. One of the core challenges in developing PI protocols for DNN inference is the substantial costs involved in implementing non-linear activation layers such as Rectified Linear Unit (ReLU). This has spawned research into the search for accurate, but efficient approximations of the ReLU function and neural architectures that operate on a stringent ReLU budget. While these methods improve efficiency and ensure data confidentiality, they often come at a significant cost to prediction accuracy. In this work, we propose a DNN architecture based on polynomial kervolution called \emph{PolyKervNet} (PKN), which completely eliminates the need for non-linear activation and max pooling layers. PolyKervNets are both FHE and MPC-friendly - they enable FHE-based encrypted inference without any approximations and improve the latency on MPC-based PI protocols without any use of garbled circuits. We demonstrate that it is possible to redesign standard convolutional neural networks (CNN) architectures such as ResNet-18 and VGG-16 with polynomial kervolution and achieve approximately $30\times$ improvement in latency with minimal loss in accuracy on many image classification tasks. </p>

Link to paper: Coming Soon

# Plaintext Results on Image Classification


## How to use
1. Clone this repository

```
git clone https://github.com/Ti-Oluwanimi/PolyKervNets.git
```

2. Pip/Conda install the following packages
    - Pytorch (v1.10, v1.11)
   ```
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```
    
    - ONNX
    ```
    pip install onnx
    ```
    
3. To run:
    - ```python main.py```

4. To reproduce our Delphi based private inference results ([paper](https://eprint.iacr.org/2020/050.pdf)):
    -   Clone the Delphi repository
    ```
    git clone https://github.com/mc2-project/delphi
    ```
    - Follow the instructions on the [Delphi](https://github.com/mc2-project/delphi) repository
