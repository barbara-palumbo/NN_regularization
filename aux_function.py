import os 
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")

def save_reconstructions_grid(
    recons,
    errors,
    param1_array,
    param2_array,
    save_dir,
    method_name="Method",
    filename="reconstruction",
    param1_name="param1",
    param2_name="param2",
    param1_fmt="{:.2e}",
    param2_fmt="{:.2e}",
):
    """
    recons: tensor (N1, N2, H, W)
    errors: tensor (N1, N2) or (N1,)
    param1_array: array-like (len N1)
    param2_array: array-like (len N2)
    """

    os.makedirs(save_dir, exist_ok=True)

    n1, n2 = recons.shape[:2]

    if errors.ndim == 1:
        errors = errors[:, None]

    min_idx = tuple(np.unravel_index(np.argmin(errors), errors.shape))

    fig, axes = plt.subplots(
        n1, n2,
        figsize=(4 * n2, 4 * n1),
        squeeze=False
    )

    for i in range(n1):
        for j in range(n2):
            ax = axes[i, j]

            im = ax.imshow(
                recons[i, j].squeeze(),
                cmap="gray"
            )

            ax.axis("off")

            title_color = "red" if (i, j) == min_idx else "black"

            title = (
                f"{param1_name}={param1_fmt.format(param1_array[i])}\n"
                f"{param2_name}={param2_fmt.format(param2_array[j])}\n"
                f"err={errors[i, j]:.2e}"
            )

            ax.set_title(title, fontsize=8, color=title_color)

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )
            cbar.ax.tick_params(labelsize=6)

    plt.suptitle(f"{method_name} reconstructions grid", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(save_dir, filename + ".png")
    plt.savefig(path, dpi=200)
    plt.close()

def plot_semi_convergence(
    alpha_array,
    errors,
    save_dir,
    method_name="Method",
    filename="semi_convergence"
):

    os.makedirs(save_dir, exist_ok=True)

    alpha_array = np.asarray(alpha_array)
    errors = np.asarray(errors)

    # Se errors è 2D (alpha x step), prendiamo il minimo su step
    if errors.ndim == 2:
        err_alpha = errors.min(axis=1)
    else:
        err_alpha = errors

    # minimo globale
    idx_best = np.argmin(err_alpha)
    best_alpha = alpha_array[idx_best]
    best_error = err_alpha[idx_best]

    plt.figure(figsize=(6, 4))
    plt.loglog(
        alpha_array,
        err_alpha,
        marker="o",
        linewidth=2,
        label=r"$\min_{\mathrm{step}} \|x_\alpha - x\|_2$"
    )

    # punto migliore
    plt.scatter(
        best_alpha,
        best_error,
        color="red",
        s=80,
        zorder=3,
        label=rf"best $\alpha={best_alpha:.2e}$"
    )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\|x_\alpha - x\|_2$")
    plt.title(
        f"{method_name} semi-convergence\n"
        rf"best $\alpha = {best_alpha:.2e}$"
    )
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, filename + ".png")
    plt.savefig(path, dpi=200)
    plt.close()

    return best_alpha, best_error

def plot_data(images, save_path, title="Data", titles=None): 
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3), squeeze=False)
    
    for i, im in enumerate(images):
        ax = axes[0, i]
        im_plot = ax.imshow(im.squeeze().cpu(), cmap="gray")
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        fig.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
        
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(save_path, f"{title}.png"), dpi=200)
    plt.close()

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)