"""Augmentation theo batch bằng affine grid."""
import math

import torch
import torch.nn.functional as F


def augment(x, rot=10.0, shift=0.1, zoom=0.1, flip=True):
    """Augmentation theo batch bằng affine grid (tương đương ImageDataGenerator của báo cáo mẫu)."""
    n = x.size(0)
    device = x.device
    ang = (torch.rand(n, device=device) * 2 - 1) * rot * math.pi / 180
    sc = 1 + (torch.rand(n, device=device) * 2 - 1) * zoom
    tx, ty = (torch.rand(n, device=device) * 2 - 1) * shift * 2, (torch.rand(n, device=device) * 2 - 1) * shift * 2
    fl = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0) if flip else torch.ones(n, device=device)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode='border', align_corners=False)
