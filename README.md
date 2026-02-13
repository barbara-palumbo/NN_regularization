NN Regularization on Compact Sets
Official implementation of the paper: "Convergence Rates for Tikhonov Regularization on Compact Sets: Application to Neural Networks" by Barbara Palumbo, Paolo Massa, and Federico Benvenuto.
This repository provides a numerical framework for solving ill-posed inverse problems (both linear and nonlinear) using a regularization strategy based on the parametrization of the solution via Multi-Layer Perceptrons (MLP). Unlike typical machine learning approaches, this method does not require training data and interprets the neural network as an implicit neural representation (learned positional encoding) defined on compact sets.
Theoretical Overview
The core of this work is a regularization method based on minimizing the Tikhonov functional on a dense sequence of compact sets. We prove that this approach achieves optimal convergence rates (e.g., O(δ 
2/3
 ) for linear operators). In the context of NNs, these compact sets are realized by networks with bounded weights.
Key Features
• Unsupervised Reconstruction: No external training datasets are required; the network is optimized directly on the data fidelity term.
• Edge Preservation: The MLP-based approach naturally provides piece-wise constant solutions and preserves sharp edges better than classical Tikhonov or Plug-and-Play (PnP) methods.
• Constraint Enforcement: Strictly enforces non-negativity by using a ReLU activation function in the final layer.
Repository Structure
• forward/: Discretized forward operators for Standard Radon and Attenuated Radon transforms.
• methods/:
    ◦ tikhonov.py: Non-negative Tikhonov regularization using an iterative projected gradient scheme.
    ◦ pnp.py: Plug-and-Play (PnP-PGD) approach using a pre-trained DnCNN denoiser.
    ◦ mlp.py: Proposed MLP-based regularization with implicit neural representations.
• test_lin_rad.py / test_non_lin_rad.py: Main scripts to run simulations on the Shepp-Logan phantom and Walnut dataset.
• aux_function.py: Utilities for visualization and metric computation (ℓ 
2
​	
  error, PSNR, SSIM).
Implementation Details
Optimization Strategy
To handle the numerical instability of neural network optimization, we adopt a robust multi-initialization protocol:
• Architecture: 4 hidden layers (Shepp-Logan) or 6 hidden layers (Walnut), with 256 neurons per layer.
• Optimizer: Adam with an initial learning rate of 1×10 
−2
  and a scheduler factor of 0.9997.
• Selection: 10 fixed random initializations are evaluated for 200 iterations; the one yielding the lowest loss is selected for a full 10,000-iteration training.
• Parameter Selection: Regularization parameter α is chosen via a two-stage oracle strategy (coarse and fine grid search) to minimize the reconstruction error.
Requirements
• Python 3.10.13
• PyTorch
• DeepInverse library (for PnP baseline)
• NumPy, Matplotlib
Citation
If you use this code in your research, please cite our paper:
@article{palumbo2024convergence,
  title={Convergence Rates for Tikhonov Regularization on Compact Sets: Application to Neural Networks},
  author={Palumbo, Barbara and Massa, Paolo and Benvenuto, Federico},
  journal={arXiv preprint},
  year={2024}
}
