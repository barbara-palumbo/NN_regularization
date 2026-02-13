import torch

class PnPPGD:
    """
    Plug-and-Play Proximal Gradient Descent (PnP-PGD).

    Solves the optimization problem:
        min_x 0.5 * ||A(x) - y||^2 + R(x)

    where R(x) is implicitly defined by a denoiser.
    """

    def __init__(
        self,
        operator,           # Operator object (e.g., RadonLinear)
        denoiser,           # Callable: denoiser(x, sigma)
        sigma_denoiser=0.01, # Noise level for the denoiser
        step_size=1e-3,      # Gradient descent step size
        n_iters=200,         # Maximum number of iterations
        tol_rel=1e-3,        # Relative convergence tolerance
        verbose=True,        # Print iteration info
    ):
        self.operator = operator
        self.denoiser = denoiser
        self.sigma_denoiser = sigma_denoiser
        self.step_size = step_size
        self.n_iters = n_iters
        self.tol_rel = tol_rel
        self.verbose = verbose

    def run(self, y, x0=None, x_gt=None):
        """
        Run the PnP-PGD algorithm.

        Parameters:
        -----------
        y : torch.Tensor
            The observed measurement tensor.
        x0 : torch.Tensor, optional
            Initial guess for the reconstruction.
        x_gt : torch.Tensor, optional
            Ground truth image (used only for printing errors).

        Returns:
        --------
        torch.Tensor
            The reconstructed image after PnP-PGD.
        """
        device = y.device
        dtype = y.dtype

        # Determine shape
        B, _, H, W = y.shape if y.ndim == 4 else (1, 1, y.shape[-2], y.shape[-1])

        # Initialization
        if x0 is None:
            x = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        else:
            x = x0.clone().to(device)

        eps = 1e-12  # small value for numerical stability

        for iteration in range(self.n_iters):
            x_old = x.detach().clone()
            x.requires_grad_(True)

            # -------- Forward operation --------
            y_pred = self.operator.forward(x)

            # -------- Data fidelity term --------
            loss = 0.5 * torch.sum((y_pred - y) ** 2)
            loss.backward()
            grad = x.grad.detach()

            # -------- Gradient descent step --------
            with torch.no_grad():
                x_tmp = x - self.step_size * grad
                x_new = self.denoiser(x_tmp, self.sigma_denoiser)
                x_new.clamp_(min=0.0)
                

            # -------- Convergence check --------
            rel_change = torch.norm(x_new - x_old) / (torch.norm(x_old) + eps)

            if self.verbose and (iteration % 50 == 0 or iteration == self.n_iters - 1):
                error_val = torch.norm(x_new - x_gt).item() if x_gt is not None else 'N/A'
                print(f"[{iteration:4d}] loss={loss.item():.4e}, rel_change={rel_change:.3e}, error={error_val}")

            if rel_change < self.tol_rel:
                if self.verbose:
                    print(f"Converged at iteration {iteration} (rel_change={rel_change:.2e})")
                break

            x = x_new.detach()

        return x.detach()