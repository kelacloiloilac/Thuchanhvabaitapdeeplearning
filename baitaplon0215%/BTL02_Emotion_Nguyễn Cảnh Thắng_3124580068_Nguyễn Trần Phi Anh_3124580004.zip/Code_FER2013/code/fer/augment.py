"""Augmentation theo batch: affine grid + random erasing."""
import math

import torch
import torch.nn.functional as F


def random_erase(x, p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3)):
    """Tô một vùng chữ nhật ngẫu nhiên bằng 0 (giá trị trung bình sau chuẩn hóa)."""
    if p <= 0:
        return x
    n, _, h, w = x.shape
    device = x.device
    area = h * w
    target = area * (scale[0] + (scale[1] - scale[0]) * torch.rand(n, device=device))
    log_ratio = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * torch.rand(n, device=device)
    aspect = torch.exp(log_ratio)
    eh = torch.sqrt(target * aspect).round().long().clamp(1, h)
    ew = torch.sqrt(target / aspect).round().long().clamp(1, w)
    y0 = (torch.rand(n, device=device) * (h - eh + 1)).long()
    x0 = (torch.rand(n, device=device) * (w - ew + 1)).long()
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    mask = (ys >= y0[:, None, None]) & (ys < (y0 + eh)[:, None, None])
    mask = mask & (xs >= x0[:, None, None]) & (xs < (x0 + ew)[:, None, None])
    mask = mask & (torch.rand(n, 1, 1, device=device) < p)
    return x.masked_fill(mask[:, None], 0.0)


def augment(x, rot=12.0, shift=0.12, zoom=0.12, flip=True,
            erase_p=0.25, erase_scale=(0.02, 0.10), erase_ratio=(0.3, 3.3)):
    """Affine (xoay/dịch/zoom/lật) + random erasing, chạy được trên CPU và CUDA."""
    n = x.size(0)
    device = x.device
    ang = (torch.rand(n, device=device) * 2 - 1) * rot * math.pi / 180
    sc = 1 + (torch.rand(n, device=device) * 2 - 1) * zoom
    tx, ty = (torch.rand(n, device=device) * 2 - 1) * shift * 2, (torch.rand(n, device=device) * 2 - 1) * shift * 2
    fl = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0) if flip else torch.ones(n, device=device)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    return random_erase(out, p=erase_p, scale=erase_scale, ratio=erase_ratio)
