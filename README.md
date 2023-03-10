# PolyKervNets: Activation-free Neural Networks For Efficient Private Inference
## Summary
<img src="https://user-images.githubusercontent.com/45424924/188140566-a2125f2e-99c8-486a-adb9-17b66578c961.png" width="850">

<p align="justify"> With the advent of cloud computing, machine learning as a service (MLaaS) has become a growing phenomenon with the potential to address many real-world problems. In an untrusted cloud environment, privacy concerns of users is a major impediment to the adoption of MLaaS. To alleviate these privacy issues and preserve data confidentiality, several private inference (PI) protocols have been proposed in recent years based on cryptographic tools like Fully Homomorphic Encryption (FHE) and Secure Multiparty Computation (MPC). Deep neural networks (DNN) have been the architecture of choice in most MLaaS deployments. One of the core challenges in developing PI protocols for DNN inference is the substantial costs involved in implementing non-linear activation layers such as Rectified Linear Unit (ReLU). This has spawned research into the search for accurate, but efficient approximations of the ReLU function and neural architectures that operate on a stringent ReLU budget. While these methods improve efficiency and ensure data confidentiality, they often come at a significant cost to prediction accuracy. In this work, we propose a DNN architecture based on polynomial kervolution called \emph{PolyKervNet} (PKN), which completely eliminates the need for non-linear activation and max pooling layers. PolyKervNets are both FHE and MPC-friendly - they enable FHE-based encrypted inference without any approximations and improve the latency on MPC-based PI protocols without any use of garbled circuits. We demonstrate that it is possible to redesign standard convolutional neural networks (CNN) architectures such as ResNet-18 and VGG-16 with polynomial kervolution and achieve approximately $30\times$ improvement in latency with minimal loss in accuracy on many image classification tasks. </p>

Link to paper: Coming Soon

# Plaintext Results on Image Classification
### Ablation (Impact of balance cp)
| Impact of cp | PKL-5 | PKA-8 | PKV-16 | PKR-18 | PKR-14 | PKR-10 | PKR-S |
|:------------:|:-----:|:-----:|:------:|:------:|:------:|:------:|:-----:|
|       0      | 65.96 | 71.64 |  81.13 |  82.1  |  83.07 |  84.22 | 75.18 |
|      0.5     | 69.55 | 80.31 |  88.91 |  89.6  |  89.84 |  91.35 | 83.38 |
|     0.75     | 69.55 | 80.07 |  89.84 |  89.53 |  89.84 |  91.17 | 83.27 |
|       1      | 69.16 | 79.97 |  90.15 |  90.31 |  90.6  |  91.88 | 84.11 |
|       2      | 64.18 | 71.44 |  80.37 |  81.17 |  82.02 |  82.26 | 74.43 |
|       3      | 62.93 | 70.11 |  78.1  |  80.21 |  80.35 |  80.3  | 71.31 |

<img src="https://user-images.githubusercontent.com/45424924/188142302-9bcbf43f-f7a1-47bc-8e31-86b4ee0347c6.png" width="850">

### CIFAR-10 Result
<img src="https://user-images.githubusercontent.com/45424924/188141151-b6ee943b-082c-4d46-8a04-175583dd8fbe.png" width="850">

# Ciphertext Results on Image Classification
### Delphi Result
|     VGG16     | Accuracy(%) | Latency(ms) | Improvement |    Resnet18   | Accuracy(%) | Latency(ms) | Improvement |
|:-------------:|:-----------:|:-----------:|:-----------:|:-------------:|:-----------:|:-----------:|:-----------:|
|    Vanilla    |    92.81    |   12018.1   |             |    Vanilla    |    72.57    |   17437.4   |             |
|   DeepReduce  |     92.6    |    6487.3   |     1.9x    |   DeepReduce  |    72.53    |    9481.1   |     1.8x    |
|  PolyApprox-1 |    88.14    |    1180.7   |    10.2x    |  PolyApprox-1 |    66.80    |    1578.3   |    11.1x    |
|  PolyApprox-2 |    87.37    |    419.7    |    28.6x    |  PolyApprox-2 |    67.45    |    606.3    |    28.8x    |
|     Square    |    80.19    |    408.5    |    29.4x    |     Square    |    61.70    |    551.4    |    31.6x    |
| PKV-16 (Ours) |     90.2    |    381.1    |    31.5x    | PKR-18 (Ours) |    70.97    |    587.9    |    29.7x    |

### TenSEAL CKKS Result
| CNN-3 Variant | Accuracy(%) | Latency(ms) | Params(M) |
|:-------------:|:-----------:|:-----------:|:---------:|
|      ReLU     |     98.5    |      NA     |   17298   |
|  PolyApprox-1 |    98.36    |    961.7    |   17298   |
|  PolyApprox-2 |    98.57    |    890.3    |   17298   |
|     Square    |    98.27    |    866.1    |   17298   |
|  PKN-3 (Ours) |    98.45    |    783.4    |    2767   |

# Future Work
Our major observation was how unstable PKNs are. We were unsuccessful in training PKN-50 (ResNet-50). PKNs are sensitive to factors such as Architecture size and complexity, Learning Rate, Polynomial degree (dp), Balance factor (cp), dataset, etc. One requires careful hyperparameter tuning to get it right. Future work in this domain include:

1.  Optimized Architectural design of PKNs for DNNs
2.  Hyperparameter Tuning for optimal performance of deep PKNs
3.  Novel and stable variants of PKNs. One direction is Pi-Nets [Chrysos et al](https://arxiv.org/abs/2003.03828)
4.  Dynamic Universal Tuning for better generations of approximate ReLU replacements.

# How to use
1. Clone this repository OR use the Polynomial Convolution by the amazing Chen Wang, the "Kervolutional Neural Network" [author](https://github.com/wang-chen/kervolution/blob/unfold/kervolution.py)

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
    ```
    python Train.py
    ```

4. To reproduce our Delphi based private inference results ([paper](https://eprint.iacr.org/2020/050.pdf)):
    -   Clone the Delphi repository
    ```
    git clone https://github.com/mc2-project/delphi
    ```
    - Follow the instructions on the [Delphi](https://github.com/mc2-project/delphi) repository
