# PolyKervNets: Activation-free Neural Networks For Efficient Private Inference
## Summary
<img src="https://user-images.githubusercontent.com/45424924/188140566-a2125f2e-99c8-486a-adb9-17b66578c961.png" width="850">

<p align="justify"> With the advent of cloud computing, machine learning as a service (MLaaS) has become a growing phenomenon with the potential to address many real-world problems. In an untrusted cloud environment, privacy concerns of users is a major impediment to the adoption of MLaaS. To alleviate these privacy issues and preserve data confidentiality, several private inference (PI) protocols have been proposed in recent years based on cryptographic tools like Fully Homomorphic Encryption (FHE) and Secure Multiparty Computation (MPC). Deep neural networks (DNN) have been the architecture of choice in most MLaaS deployments. One of the core challenges in developing PI protocols for DNN inference is the substantial costs involved in implementing non-linear activation layers such as Rectified Linear Unit (ReLU). This has spawned research into the search for accurate, but efficient approximations of the ReLU function and neural architectures that operate on a stringent ReLU budget. While these methods improve efficiency and ensure data confidentiality, they often come at a significant cost to prediction accuracy. In this work, we propose a DNN architecture based on polynomial kervolution called \emph{PolyKervNet} (PKN), which completely eliminates the need for non-linear activation and max pooling layers. PolyKervNets are both FHE and MPC-friendly - they enable FHE-based encrypted inference without any approximations and improve the latency on MPC-based PI protocols without any use of garbled circuits. We demonstrate that it is possible to redesign standard convolutional neural networks (CNN) architectures such as ResNet-18 and VGG-16 with polynomial kervolution and achieve approximately $30\times$ improvement in latency with minimal loss in accuracy on many image classification tasks. </p>

# Results
Check Paper: [IEEE](https://ieeexplore.ieee.org/abstract/document/10136177), [OpenReview](https://openreview.net/pdf?id=OGzt9NKC0lO), [ResearchGate](https://www.researchgate.net/publication/371230167_PolyKervNets_Activation-free_Neural_Networks_For_Efficient_Private_Inference)

Update: A new version of PolyKervNets is at [Cryptology](https://eprint.iacr.org/2023/1917)

# How to use
1. Clone this repository.
2. Create a conda environment, and install these main packages:
    - Python >= 3.8
    - [PyTorch](https://pytorch.org/get-started/locally/)
    - [MoMo](https://github.com/fabian-sp/MoMo/tree/main)
3. To run, go through the ```train.py``` file and make desired changes based on the experiment you want to run. Note that this code is not production friendly, but we have made it quite easy to navigate.

# Future Work
Our major observation was how unstable PKNs are. We were unsuccessful in training PKN-50 (ResNet-50) (RPKNs was successful). PKNs are sensitive to factors such as Architecture size and complexity, Learning Rate, Polynomial degree (dp), Balance factor (cp), dataset, etc. One requires careful hyperparameter tuning to get it right.

### Update: We were able to build a more stable version of PKNs. New future directions are listed below:

1. Investigating the potential benefits of combining R-PKNs with gradient clipping to determine if this approach can yield comparable or superior results in terms of stability and overall performance.
2. Exploring layer-wise learning rate initialization, where deeper layers are assigned different learning rates than initial layers, in order to further optimize the training process for polynomial-based networks. A quick experiment with this gave RPKR-50 an accuracy of 87.9\% without requiring tuning or knowledge distillation.
3. Exploring alternative optimization techniques, such as Quasi-Newton based approaches, to determine if certain types of optimizers exhibit superior performance and convergence properties when applied to polynomial-based networks.
4. Extending the scope of our conclusions to assess whether they are applicable to other polynomial-based approaches, beyond RPKRs, in various deep learning scenarios.
5. Evaluating the generalizability of our approach to different datasets and model architectures, such as Vision Transformers (ViTs), to determine its effectiveness in a broader context.
