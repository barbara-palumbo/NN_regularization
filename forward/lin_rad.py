import torch
import torch.nn.functional as F


def linear_radon(images, angles):
    """
    Linear Radon transform using rotation + line integration.

    images: (B, 1, H, W)
    angles: (num_angles,) in degrees
    returns: (B, num_angles, W)
    """
    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype
    num_angles = len(angles)

    angles_rad = -angles * torch.pi / 180.0
    angles_rad = angles_rad.to(device=device, dtype=dtype)

    # ----------------- Rotation matrices -----------------
    cos_t = torch.cos(angles_rad)
    sin_t = torch.sin(angles_rad)

    R = torch.zeros(num_angles, 2, 3, device=device, dtype=dtype)
    R[:, 0, 0] = cos_t
    R[:, 0, 1] = sin_t
    R[:, 1, 0] = -sin_t
    R[:, 1, 1] = cos_t

    R = R.unsqueeze(0).expand(B, -1, -1, -1)
    R_flat = R.reshape(B * num_angles, 2, 3)

    images_flat = images.unsqueeze(1).expand(-1, num_angles, -1, -1, -1)
    images_flat = images_flat.reshape(B * num_angles, C, H, W)

    grids = F.affine_grid(R_flat, images_flat.size(), align_corners=True)
    rotated = F.grid_sample(
        images_flat, grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    rotated = rotated.reshape(B, num_angles, C, H, W)

    sinogram = rotated[:, :, 0].sum(dim=2) 

    return sinogram


def linear_radon_adjoint(sino, angles, img_shape):
    """
    Computes A* g via autograd.

    sino: (B, num_angles, W)
    returns: (B, 1, H, W)
    """
    B = sino.shape[0]
    H, W = img_shape

    x = torch.zeros(
        (B, 1, H, W),
        device=sino.device,
        dtype=sino.dtype,
        requires_grad=True
    )

    Ax = linear_radon(x, angles)
    loss = (Ax * sino).sum()
    loss.backward()

    return x.grad


class LinearRadon:
    def __init__(self, angles):
        self.angles = angles

    def forward(self, x):
        return linear_radon(x, self.angles)

    def adjoint(self, g, img_shape):
        return linear_radon_adjoint(g, self.angles, img_shape)