"""Chọn thiết bị huấn luyện (CPU/CUDA)."""
import torch


def get_device(name='cpu'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)
