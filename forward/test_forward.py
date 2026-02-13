import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

from lin_rad import RadonLinear



img_size_pha = 128
img_size_torch = 256
n_angles = 180

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "forward/comparison"
os.makedirs(save_dir, exist_ok=True)

phantom = shepp_logan_phantom()
phantom = resize(
    phantom,
    (img_size_pha, img_size_pha),
    mode="reflect",
    anti_aliasing=True
)
phantom = phantom.astype(np.float32)

phantom_t = shepp_logan_phantom()
phantom_t = resize(
    phantom_t,
    (img_size_torch, img_size_torch),
    mode="reflect",
    anti_aliasing=True
)
x_torch = torch.tensor(phantom_t, device=device).unsqueeze(0).unsqueeze(0)







fig, axes = plt.subplots(1, 2, figsize=(15, 4))
im = axes[0].imshow(phantom, cmap="gray", aspect="auto")
axes[0].set_title("phantom (skimage)")
axes[0].axis("off")
plt.colorbar(im, ax=axes[0], fraction=0.046)
im = axes[1].imshow(x_torch.squeeze().detach().cpu().numpy(), cmap="gray", aspect="auto")
axes[1].set_title("phantom (torch)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "phantom.png"),
    dpi=200
)
plt.close()


# ===================== ANGLES =====================
angles = np.linspace(0., 180., n_angles, endpoint=False)
angles_torch = torch.tensor(angles, device=device)

sino_physics = radon(
    phantom,
    theta=angles,
    circle=True
)
sino_physics = sino_physics.T  
sino_physics = sino_physics / np.max(sino_physics)

# ===================== CUSTOM RADON =====================
A = RadonLinear(angles_torch)
sino_custom = A.forward(x_torch)  # (1, num_angles, W)
sino_custom = sino_custom.squeeze(0).detach().cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
im = ax.imshow(sino_custom, cmap="gray", aspect="auto")
ax.set_title("Custom Radon operator")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "sino_torch.png"),
    dpi=200
)
plt.close()

sino_custom = np.flip(sino_custom, axis=(0, 1))
print(sino_custom.shape, sino_physics.shape)


sino_custom = resize(sino_custom, (180, 128), order=1, mode='reflect', anti_aliasing=True)
sino_custom = sino_custom / np.max(sino_custom)

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
im = ax.imshow(sino_custom, cmap="gray", aspect="auto")
ax.set_title("Custom Radon operator")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "sino_torch_resize.png"),
    dpi=200
)
plt.close()


# # ===================== MATCH SHAPES =====================
# # skimage usa H come detector axis
# min_width = min(sino_physics.shape[1], sino_custom.shape[1])
# sino_physics = sino_physics[:, :min_width]
# sino_custom = sino_custom[:, :min_width]
# sino_custom = np.flip(sino_custom, axis=(0, 1))
# print(sino_custom.shape)

# # ===================== ERROR =====================
diff = sino_custom - sino_physics
error_l2 = np.linalg.norm(diff)

# # ===================== SAVE FIGURE =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im = axes[0].imshow(sino_physics, cmap="gray", aspect="auto")
axes[0].set_title("Physics-based Radon (skimage)")
axes[0].axis("off")
plt.colorbar(im, ax=axes[0], fraction=0.046)

im = axes[1].imshow(sino_custom, cmap="gray", aspect="auto")
axes[1].set_title("Custom Radon operator")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

im = axes[2].imshow(diff, cmap="seismic", aspect="auto")
axes[2].set_title("Difference (custom - physics)")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "radon_comparison.png"),
    dpi=200
)
plt.close()

# ===================== SAVE METRICS =====================
txt_path = os.path.join(save_dir, "comparison_metrics.txt")

# with open(txt_path, "w") as f:
#     f.write("Radon operator comparison\n")
#     f.write("===========================\n\n")
#     f.write(f"Image size: {img_size} x {img_size}\n")
#     f.write(f"Number of angles: {n_angles}\n")
#     f.write("Reference operator: skimage.transform.radon\n")
#     f.write("Custom operator: RadonLinear\n\n")
#     f.write(f"L2 norm of sinogram difference: {error_l2:.6e}\n")

# print("✓ Comparison completed")
# print(f"✓ Figure saved in: {save_dir}/radon_comparison.png")
# print(f"✓ Metrics saved in: {save_dir}/comparison_metrics.txt")