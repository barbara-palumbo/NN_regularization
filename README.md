# Convergence rates for Tikhonov regularization on compact sets: application to neural networks

## Overview

This repository contains the source code for the numerical experiments presented in [[1]](https://arxiv.org/abs/2505.19936). The paper introduces a novel regularization method for inverse problems, where the solution is parameterized using a Multi-Layer Perceptron (MLP).

---

## Numerical Experiments

The code performs reconstructions for inverse problems involving:

- Standard (linear) Radon transform  
- Non-linear attenuated Radon transform  

The results are compared using the following approaches:

- **Non-negative Tikhonov regularization**
- **Plug-and-Play reconstruction (PnP-PGD)**
- **Our proposed MLP-based regularization method**
___

## How to run our code

The tests concerning the inverse problem with standard (linear) and (non-linear) attenuated Radon trasform can be performed by running `test_lin_rad.py` and `test_non_lin_rad.py`, respectively.

***
## References
[1] Palumbo, B., Massa, P., & Benvenuto, F. (2025). Convergence rates for Tikhonov regularization on compact sets: application to neural networks. arXiv preprint arXiv:2505.19936.
