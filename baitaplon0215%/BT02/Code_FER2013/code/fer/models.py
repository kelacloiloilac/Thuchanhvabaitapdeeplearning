"""Kiến trúc CNN cho FER2013."""
import torch.nn as nn

from .config import ARCH


def conv_block(cin, cout, n, bn=True, drop=0.2):
    L = []
    for k in range(n):
        L.append(nn.Conv2d(cin if k == 0 else cout, cout, 3, padding=1))
        if bn:
            L.append(nn.BatchNorm2d(cout))
        L.append(nn.ReLU(inplace=True))
    L += [nn.MaxPool2d(2), nn.Dropout(drop)]
    return L


class FERNet(nn.Module):
    """CNN: các khối [Conv-(BN)-ReLU]xn + MaxPool + Dropout, rồi Flatten-Dense-Dropout-Dense(7)."""

    def __init__(self, channels=(16, 32, 64), reps=(1, 1, 1), dense=256, bn=True, drop_conv=0.2, drop_fc=0.5):
        super().__init__()
        L, c = [], 1
        for ch, n in zip(channels, reps):
            L += conv_block(c, ch, n, bn, drop_conv)
            c = ch
        self.features = nn.Sequential(*L)
        side = 48 // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * side * side, dense),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),
            nn.Linear(dense, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch='small', **overrides):
    cfg = dict(ARCH[arch])
    cfg.update(overrides)
    return FERNet(**cfg)
