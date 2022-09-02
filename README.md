# PolyKervNets: Activation-free Neural Networks For Efficient Private Inference

<p align="justify"> With the advent of cloud computing, machine learning as a service (MLaaS) has become a growing phenomenon with the potential to address many real-world problems. In an untrusted cloud environment, privacy concerns of users is a major impediment to the adoption of MLaaS. To alleviate these privacy issues and preserve data confidentiality, several private inference (PI) protocols have been proposed in recent years based on cryptographic tools like Fully Homomorphic Encryption (FHE) and Secure Multiparty Computation (MPC). Deep neural networks (DNN) have been the architecture of choice in most MLaaS deployments. One of the core challenges in developing PI protocols for DNN inference is the substantial costs involved in implementing non-linear activation layers such as Rectified Linear Unit (ReLU). This has spawned research into the search for accurate, but efficient approximations of the ReLU function and neural architectures that operate on a stringent ReLU budget. While these methods improve efficiency and ensure data confidentiality, they often come at a significant cost to prediction accuracy. In this work, we propose a DNN architecture based on polynomial kervolution called \emph{PolyKervNet} (PKN), which completely eliminates the need for non-linear activation and max pooling layers. PolyKervNets are both FHE and MPC-friendly - they enable FHE-based encrypted inference without any approximations and improve the latency on MPC-based PI protocols without any use of garbled circuits. We demonstrate that it is possible to redesign standard convolutional neural networks (CNN) architectures such as ResNet-18 and VGG-16 with polynomial kervolution and achieve approximately $30\times$ improvement in latency with minimal loss in accuracy on many image classification tasks. </p>

Link to paper: Coming Soon

# Plaintext Results on Image Classification
## Ablation
| Impact of cp | PKL-5 | PKA-8 | PKV-16 | PKR-18 | PKR-14 | PKR-10 | PKR-S |
|:------------:|:-----:|:-----:|:------:|:------:|:------:|:------:|:-----:|
|       0      | 65.96 | 71.64 |  81.13 |  82.1  |  83.07 |  84.22 | 75.18 |
|      0.5     | 69.55 | 80.31 |  88.91 |  89.6  |  89.84 |  91.35 | 83.38 |
|     0.75     | 69.55 | 80.07 |  89.84 |  89.53 |  89.84 |  91.17 | 83.27 |
|       1      | 69.16 | 79.97 |  90.15 |  90.31 |  90.6  |  91.88 | 84.11 |
|       2      | 64.18 | 71.44 |  80.37 |  81.17 |  82.02 |  82.26 | 74.43 |
|       3      | 62.93 | 70.11 |  78.1  |  80.21 |  80.35 |  80.3  | 71.31 |

## How to use
1. Clone this repository or use the Polynomial Convolution by the original [author](https://github.com/wang-chen/kervolution/blob/unfold/kervolution.py)

```
git clone https://github.com/Ti-Oluwanimi/PolyKervNets.git
```

2. Pip/Conda install the following packages
    - Pytorch (v1.10, v1.11)
   ```
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```
    
    - ONNX: You will need this in Delphi if you are using pytorch
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
