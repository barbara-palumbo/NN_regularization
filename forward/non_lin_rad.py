import torch
import torch.nn.functional as F


def non_linear_radon(images, angles, par_cum=1.0):
    """
    Non-linear Radon transform with exponential attenuation.

    images: (B, 1, H, W)
    angles: (num_angles,) in degrees
    returns: (B, num_angles, W)
    """
    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype
    num_angles = len(angles)

    angles_rad = -angles * torch.pi / 180.0
    angles_rad = angles_rad.to(device=device, dtype=dtype)

    # ----------------- Rotation -----------------
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
        padding_mode="border",
        align_corners=True
    )

    rotated = rotated.reshape(B, num_angles, C, H, W)

    # ----------------- Non-linear line integral -----------------
    samples = rotated[:, :, 0]  # (B, num_angles, H, W)
    cumulative = torch.cumsum(samples, dim=2)
    attenuation = torch.exp(-par_cum * cumulative)

    sinogram = torch.sum(samples * attenuation, dim=2)

    return sinogram


    
class NonLinearRadon:
    def __init__(self, angles, par_cum=1.0):
        self.angles = angles
        self.par_cum = par_cum

    def forward(self, x):
        return non_linear_radon(x, self.angles, self.par_cum)