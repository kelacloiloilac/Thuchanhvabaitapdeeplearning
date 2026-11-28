"""Trực quan hóa kernel, activation maximization và feature map."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def conv_layers(model):
    return [m for m in model.features if isinstance(m, nn.Conv2d)]


def show_kernels(conv, path, title, nmax=32):
    W = conv.weight.detach().numpy()
    if W.shape[1] == 1:
        fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < W.shape[0]:
                ax.imshow(W[k, 0], cmap='bwr', vmin=-abs(W).max(), vmax=abs(W).max())
    else:
        M = W.mean(1)
        fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < min(nmax, M.shape[0]):
                ax.imshow(M[k], cmap='bwr', vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def feature_extractor(model):
    last = conv_layers(model)[-1]
    idx_last = list(model.features).index(last)
    has_bn = idx_last + 2 < len(model.features) and isinstance(model.features[idx_last + 1], nn.BatchNorm2d)
    upper = idx_last + 3 if has_bn else idx_last + 2
    sub = nn.Sequential(*list(model.features)[:upper])
    for p_ in sub.parameters():
        p_.requires_grad_(False)
    sub.eval()
    return sub


def activation_maximization(sub, path):
    torch.manual_seed(0)
    nf = 32
    x = (torch.randn(nf, 1, 48, 48) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        a = sub(x)
        loss = -a[torch.arange(nf), torch.arange(nf)].mean() + 1e-3 * (x ** 2).mean()
        loss.backward()
        opt.step()
    fig, axs = plt.subplots(4, 8, figsize=(11, 6))
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(x[k, 0].detach().numpy(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'f{k}', fontsize=7)
    fig.suptitle('Activation maximization: mẫu đầu vào kích hoạt mạnh nhất 32 filter đầu của conv cuối')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_feature_maps(sub, Xt, Xte, i0, path):
    fm = sub(Xt[i0:i0 + 1])[0].detach().numpy()
    fig, axs = plt.subplots(4, 9, figsize=(12, 5.6))
    axs[0, 0].imshow(Xte[i0], cmap='gray')
    axs[0, 0].set_title('Input (Happy)', fontsize=7)
    for ax in axs.ravel():
        ax.axis('off')
    for k in range(31):
        axs.ravel()[k + 1].imshow(fm[k], cmap='viridis')
        axs.ravel()[k + 1].set_title(f'f{k}', fontsize=6)
    fig.suptitle('Feature map của conv cuối (6x6) cho một ảnh Happy')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
