import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, num_layers, num_nodes_per_layer, activation='leakyrelu', input_dim=2):
        """
        MLP mapping 2D spatial coordinates to a scalar image intensity.

        Parameters
        ----------
        num_layers : int
            Total number of layers (including input and output).
        num_nodes_per_layer : int
            Number of hidden units per layer.
        activation : str
            Activation function for hidden layers ('leakyrelu' or 'sigmoid').
        """
        super(MLP, self).__init__()

        layers = []
        
        # Input layer: (x, y) -> hidden representation
        layers.append(nn.Linear(input_dim, num_nodes_per_layer)) 
        if activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        # Hidden layers
        for _ in range(1, num_layers - 1): 
            if activation == 'leakyrelu':
                layers.append(nn.Linear(num_nodes_per_layer, num_nodes_per_layer))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'sigmoid':
                layers.append(nn.Linear(num_nodes_per_layer, num_nodes_per_layer))
                layers.append(nn.Sigmoid())

        # Output layer: scalar intensity
        layers.append(nn.Linear(num_nodes_per_layer, 1))  
        layers.append(nn.ReLU())  # positivity constraint

        self.network = nn.Sequential(*layers)

        # Xavier initialization for linear layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Weight initialization.
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad(): 
                # nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
                #nn.init.xavier_uniform_(m.bias) 

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Output intensities of shape (N, 1).
        """
        return self.network(x)
    



class ImplicitTikhonovSolver:
    """
    Tikhonov-regularized implicit reconstruction with an INR (MLP).

    Solves:
        min_theta ||A(f_theta(points)) - y||^2 + alpha ||f_theta(points)||^2
    """

    def __init__(
        self,
        model,
        operator,
        optimizer,
        alpha=1e-4,
        n_iters=5000,
        scheduler_gamma=0.99,
        lr_min=5e-5,
        verbose=True,
        device="cuda",
    ):
        self.model = model.to(device)
        self.operator = operator
        self.optimizer = optimizer
        self.alpha = alpha
        self.n_iters = n_iters
        self.lr_min = lr_min
        self.verbose = verbose
        self.device = device

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=scheduler_gamma
        )

        self.mse = nn.MSELoss()

    def run(self, points, y, image_gt=None):
        """
        Run the optimization.

        Parameters
        ----------
        n_iters_override : int, optional
            If provided, overrides self.n_iters
        """

        best_loss = float("inf")
        best_state = None
        loss_history = []

        H, W = image_gt.shape[-2:] if image_gt is not None else None

        for it in range(self.n_iters):

            img = self.model(points).view(1, 1, H, W)
            pred = self.operator.forward(img)

            loss = self.mse(pred.view(-1), y.view(-1)) \
                + self.alpha * torch.mean(img ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # LR scheduling
            lr = self.optimizer.param_groups[0]["lr"]
            if lr > self.lr_min:
                self.scheduler.step()
            else:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr_min

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {
                    k: v.detach().clone()
                    for k, v in self.model.state_dict().items()
                }

            if self.verbose and it % 100 == 0:
                print(f"[Iter {it:5d}] Loss={loss_val:.3e}, LR={lr:.2e}")

        # carica i pesi migliori
        self.model.load_state_dict(best_state)

        return self.model, best_state, loss_history, best_loss
