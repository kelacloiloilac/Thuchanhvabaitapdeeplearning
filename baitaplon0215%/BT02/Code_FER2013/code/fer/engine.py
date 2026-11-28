"""Vòng huấn luyện, đánh giá và dự đoán dùng chung."""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def make_optimizer(model, opt, lr, wd):
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)


def make_scheduler(opt, sched, max_lr, total_steps):
    if sched == 'cosine':
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)
    return None


@torch.no_grad()
def predict(model, X, bs=512, device=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        batch = X[i:i + bs]
        if device is not None:
            batch = batch.to(device)
        out.append(model(batch).cpu())
    return torch.cat(out)


def train_one_epoch(model, opt, sched, X, y, bs, do_aug, ls, device=None):
    model.train()
    perm = torch.randperm(len(X))
    tl = tc = 0
    for i in range(0, len(X), bs):
        idx = perm[i:i + bs]
        x, yb = X[idx], y[idx]
        if device is not None:
            x, yb = x.to(device), yb.to(device)
        if do_aug:
            x = augment(x)
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, yb, label_smoothing=ls)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        tl += loss.item() * len(yb)
        tc += (out.argmax(1) == yb).sum().item()
    return tl / len(X), tc / len(X)


def evaluate(model, X, y, device=None):
    logits = predict(model, X, device=device)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def fit(model, opt, sched, Xtr, ytr, Xva, yva, epochs, bs, do_aug, ls, tag, device=None, t0=None, log=print):
    if t0 is None:
        t0 = time.time()
    hist, best = [], dict(val_acc=0, epoch=-1)
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, opt, sched, Xtr, ytr, bs, do_aug, ls, device=device)
        val_loss, val_acc = evaluate(model, Xva, yva, device=device)
        r = dict(epoch=ep + 1, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                 lr=opt.param_groups[0]['lr'], elapsed_min=(time.time() - t0) / 60)
        hist.append(r)
        log(json.dumps(r), flush=True)
        if val_acc > best['val_acc']:
            best = dict(val_acc=val_acc, val_loss=val_loss, epoch=ep + 1)
            torch.save(model.state_dict(), tag + '.pt')
    return hist, best


def load_checkpoint(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device if device is not None else 'cpu'))
    return model
