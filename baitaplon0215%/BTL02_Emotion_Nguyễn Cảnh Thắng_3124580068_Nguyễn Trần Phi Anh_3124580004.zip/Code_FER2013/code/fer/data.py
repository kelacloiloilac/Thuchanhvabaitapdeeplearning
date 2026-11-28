"""Nạp dữ liệu FER2013 đã tiền xử lý (.npz) và chuyển sang tensor."""
import numpy as np
import torch


def load_fer(path):
    d = np.load(path, allow_pickle=True)
    X, y, u = d['X'], d['y'], d['usage']
    sp = lambda name: (X[u == name], y[u == name])
    return sp('Training'), sp('PublicTest'), sp('PrivateTest')


def to_tensor(X):
    return (torch.from_numpy(X).float().unsqueeze(1) / 255.0 - 0.5) / 0.5


def load_splits(path):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer(path)
    Xtr, Xva, Xte = map(to_tensor, (Xtr, Xva, Xte))
    ytr, yva, yte = map(torch.from_numpy, (ytr, yva, yte))
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
