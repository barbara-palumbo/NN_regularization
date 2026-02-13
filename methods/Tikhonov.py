import torch

class TikhonovLandweber:
    """
    Tikhonov-regularized Landweber reconstruction.

    Solves:
        min_x 0.5 * ||A(x) - y||^2 + 0.5 * alpha * ||x||^2
    """

    def __init__(
        self,
        operator,          # forward operator object (e.g. RadonLinear)
        alpha=0.05,        # Tikhonov regularization parameter
        step_size=1e-3,    # Landweber step size
        n_iters=200,       # maximum number of iterations
        tol_rel=1e-3,      # relative convergence tolerance
        par_cum=1.0,       # used only for nonlinear Radon
        verbose=True
    ):
        self.operator = operator
        self.alpha = alpha
        self.step_size = step_size
        self.n_iters = n_iters
        self.tol_rel = tol_rel
        self.par_cum = par_cum
        self.verbose = verbose

    def run(self, y, x0=None):
        device = y.device
        dtype = y.dtype

        # infer image size
        if y.ndim == 4:
            B, _, H, W = y.shape
        else:
            B, H, W = 1, y.shape[-2], y.shape[-1]

        # initialization
        if x0 is None:
            x = torch.zeros((B, 1, W, W), device=device, dtype=dtype)
        else:
            x = x0.clone().to(device)

        eps = 1e-12

        for it in range(self.n_iters):
            x.requires_grad_(True)

            # forward operator
            y_pred = self.operator.forward(x)

            # Tikhonov objective
            loss = (
                0.5 * torch.sum((y_pred - y) ** 2)
                + 0.5 * self.alpha * torch.sum(x ** 2)
            )
            loss.backward()

            grad_x = x.grad.detach()

            # Landweber update
            with torch.no_grad():
                x_new = x - self.step_size * grad_x
                x_new.clamp_(min=0.0)

            # relative change
            rel_change = torch.norm(x_new - x) / (torch.norm(x) + eps)

            if self.verbose and (it % 50 == 0 or it == self.n_iters - 1):
                print(
                    f"[{it:4d}] "
                    f"loss={loss.item():.4e}, "
                    f"rel_change={rel_change:.3e}"
                )

            if rel_change < self.tol_rel:
                if self.verbose:
                    print(
                        f"Converged at iteration {it} "
                        f"(rel_change={rel_change:.2e})"
                    )
                break

            x = x_new.detach()

        return x.detach()