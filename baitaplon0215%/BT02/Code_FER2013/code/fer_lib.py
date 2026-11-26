"""Thư viện dùng chung: dữ liệu, augmentation, mô hình CNN cho FER2013 (PyTorch)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # đúng thứ tự nhãn của đề (0..6)

def load_fer(path):
    d = np.load(path, allow_pickle=True)
    X, y, u = d['X'], d['y'], d['usage']
    sp = lambda name: (X[u == name], y[u == name])
    return sp('Training'), sp('PublicTest'), sp('PrivateTest')

def to_tensor(X):  # uint8 (N,48,48) -> float (N,1,48,48), chuẩn hóa ~[-1,1]
    return (torch.from_numpy(X).float().unsqueeze(1) / 255.0 - 0.5) / 0.5

def augment(x, rot=10.0, shift=0.1, zoom=0.1, flip=True):
    """Augmentation theo batch bằng affine grid (tương đương ImageDataGenerator của báo cáo mẫu)."""
    n = x.size(0)
    ang = (torch.rand(n) * 2 - 1) * rot * math.pi / 180
    sc = 1 + (torch.rand(n) * 2 - 1) * zoom
    tx, ty = (torch.rand(n) * 2 - 1) * shift * 2, (torch.rand(n) * 2 - 1) * shift * 2
    fl = torch.where(torch.rand(n) < 0.5, -1.0, 1.0) if flip else torch.ones(n)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode='border', align_corners=False)

def conv_block(cin, cout, n, bn=True, drop=0.2):
    L = []
    for k in range(n):
        L.append(nn.Conv2d(cin if k == 0 else cout, cout, 3, padding=1))
        if bn: L.append(nn.BatchNorm2d(cout))
        L.append(nn.ReLU(inplace=True))
    L += [nn.MaxPool2d(2), nn.Dropout(drop)]
    return L

class FERNet(nn.Module):
    """CNN: các khối [Conv-(BN)-ReLU]xn + MaxPool + Dropout, rồi Flatten-Dense-Dropout-Dense(7)."""
    def __init__(self, channels=(16, 32, 64), reps=(1, 1, 1), dense=256, bn=True, drop_conv=0.2, drop_fc=0.5):
        super().__init__()
        L, c = [], 1
        for ch, n in zip(channels, reps):
            L += conv_block(c, ch, n, bn, drop_conv); c = ch
        self.features = nn.Sequential(*L)
        side = 48 // (2 ** len(channels))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c * side * side, dense), nn.ReLU(inplace=True),
                                        nn.Dropout(drop_fc), nn.Linear(dense, 7))
    def forward(self, x):
        return self.classifier(self.features(x))

# Cấu hình các mô hình dùng trong thí nghiệm
ARCH = {
    'small':  dict(channels=(16, 32, 64),  reps=(1, 1, 1)),
    'medium': dict(channels=(32, 64, 128), reps=(1, 2, 2)),
    'small_nobn': dict(channels=(16, 32, 64), reps=(1, 1, 1), bn=False),
}

@torch.no_grad()
def predict(model, X, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs]))
    return torch.cat(out)
