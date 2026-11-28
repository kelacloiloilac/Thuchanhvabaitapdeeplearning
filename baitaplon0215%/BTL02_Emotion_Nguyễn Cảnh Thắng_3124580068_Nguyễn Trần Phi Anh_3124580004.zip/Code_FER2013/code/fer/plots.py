"""Vẽ đường cong, confusion matrix và các hình dữ liệu."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .augment import augment
from .config import CLASSES
from .data import to_tensor


def save_compare_curves(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in runs:
        e = [h['epoch'] for h in r['hist']]
        ax[0].plot(e, [h['val_acc'] for h in r['hist']],
                   label=r['name'].split('_')[0] + ' ' + '_'.join(r['name'].split('_')[1:]))
        ax[1].plot(e, [h['val_loss'] for h in r['hist']])
    ax[0].set_title('Validation accuracy (PublicTest)')
    ax[1].set_title('Validation loss')
    for x in ax:
        x.set_xlabel('epoch')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_run_curves(run, path):
    h = run['hist']
    e = [x['epoch'] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(e, [x['train_acc'] for x in h], label='train')
    ax[0].plot(e, [x['val_acc'] for x in h], label='val')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[1].plot(e, [x['train_loss'] for x in h], label='train')
    ax[1].plot(e, [x['val_loss'] for x in h], label='val')
    ax[1].set_title('Loss')
    ax[1].legend()
    for x in ax:
        x.grid(alpha=.3)
        x.set_xlabel('epoch')
    fig.suptitle(run['name'])
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_confusion_matrix(cm, name, acc, path):
    cmn = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(cmn, cmap='Blues')
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{cmn[i, j]:.0f}%', ha='center', va='center',
                    color='white' if cmn[i, j] > 50 else 'black', fontsize=9)
    ax.set_title(f"Confusion matrix – {name} (PrivateTest acc {acc*100:.2f}%)", fontsize=10)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_case_grid(X, y_true, y_pred, idx, path, title):
    fig, axs = plt.subplots(4, 8, figsize=(12, 6.8))
    for k, ax in enumerate(axs.ravel()):
        ax.axis('off')
        if k < len(idx):
            i = idx[k]
            ax.imshow(X[i], cmap='gray')
            ax.set_title(f'{CLASSES[y_pred[i]]} ({CLASSES[y_true[i]]})', fontsize=8,
                         color='green' if y_pred[i] == y_true[i] else 'red')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def save_class_hist(ytr, yva, yte, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax_, (name, y) in zip(ax, [('Training', ytr), ('PublicTest', yva), ('PrivateTest', yte)]):
        ax_.bar(CLASSES, np.bincount(y, minlength=7))
        ax_.set_title(f'{name} ({len(y)})')
        ax_.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_samples(Xtr, ytr, path):
    fig, axs = plt.subplots(2, 7, figsize=(12, 4))
    for c in range(7):
        ids = np.where(ytr == c)[0][:2]
        for r_, i in enumerate(ids):
            axs[r_, c].imshow(Xtr[i], cmap='gray')
            axs[r_, c].axis('off')
            if r_ == 0:
                axs[r_, c].set_title(CLASSES[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_augmentation(Xtr, path):
    xs = to_tensor(Xtr[:8])
    fig, axs = plt.subplots(3, 8, figsize=(12, 4.8))
    torch.manual_seed(1)
    for r_ in range(3):
        xa = xs if r_ == 0 else augment(xs)
        for c in range(8):
            axs[r_, c].imshow(xa[c, 0], cmap='gray')
            axs[r_, c].axis('off')
    axs[0, 0].set_title('gốc', fontsize=8)
    axs[1, 0].set_title('augment', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
