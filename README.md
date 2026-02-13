:::writing{variant=“standard” id=“58342”}

NN Regularization with Neural Networks for Inverse Problems

This repository contains the implementation of several reconstruction methods for inverse problems based on Radon operators, including classical variational techniques, Plug-and-Play priors, and a neural implicit regularization approach.

The code accompanies the numerical experiments presented in our work on neural-network-based regularization methods.

⸻

Repository Structure

forward/            Discretized forward operators for Standard and Attenuated Radon transforms

methods/
    tikhonov.py     Non-negative Tikhonov regularization via projected gradient descent
    pnp.py          Plug-and-Play (PnP-PGD) reconstruction with a pretrained DnCNN denoiser
    mlp.py          Proposed MLP-based implicit neural regularization

test_lin_rad.py     Experiments with the Standard Radon operator
test_non_lin_rad.py Experiments with the Attenuated Radon operator

aux_function.py     Visualization utilities and reconstruction metrics (PSNR, SSIM, L2 error)

mplemented Methods

The repository compares three reconstruction approaches:
	•	Non-negative Tikhonov regularization
	•	Plug-and-Play reconstruction (PnP-PGD)
	•	Implicit neural regularization using MLPs

The neural approach represents the reconstruction as a continuous function learned through coordinate-based neural networks.

Requirements

The code was tested with:
	•	Python 3.10.13
	•	PyTorch
	•	DeepInverse (for the Plug-and-Play baseline)
	•	NumPy
	•	Matplotlib
