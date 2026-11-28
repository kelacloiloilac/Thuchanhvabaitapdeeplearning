"""Chạy thí nghiệm, lưu kết quả và đọc lại nhật ký."""
import csv
import glob
import json
import os
import time

import torch

from .config import DATA_NPZ, RUNS_DIR
from .data import load_splits
from .device import get_device
from .engine import fit, load_checkpoint, make_optimizer, make_scheduler, predict, seed_everything
from .models import build_model

EXPERIMENTS = [
    dict(name='E1_small_noaug', arch='small', aug=0, epochs=12, long_epochs=20),
    dict(name='E2_small_aug', arch='small', aug=1, epochs=12, long_epochs=25),
    dict(name='E3_small_aug_onecycle', arch='small', aug=1, epochs=12, long_epochs=25,
         sched='cosine', lr=3e-3),
    dict(name='E4_small_nobn_aug', arch='small_nobn', aug=1, epochs=12, long_epochs=25),
    dict(name='E5_small_sgd_onecycle', arch='small', aug=1, epochs=12, long_epochs=25,
         opt='sgd', sched='cosine', lr=0.05, wd=5e-4),
    dict(name='E6_medium_aug_onecycle', arch='medium', aug=1, epochs=10, long_epochs=30,
         sched='cosine', lr=3e-3, ls=0.1),
]


def load_runs(runs_dir=RUNS_DIR):
    runs = [json.load(open(path)) for path in glob.glob(os.path.join(runs_dir, 'E*.json'))]
    return sorted(runs, key=lambda r: r['name'])


def best_run(runs):
    return max(runs, key=lambda r: r['best']['val_acc'])


def write_experiment_log(runs, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Thí nghiệm', 'Kiến trúc', 'Số tham số', 'Augmentation', 'Optimizer', 'LR', 'Scheduler',
                    'Weight decay', 'Label smoothing', 'Class weights', 'Epochs', 'Best epoch',
                    'Train acc @best', 'Val loss @best', 'Val acc (PublicTest)', 'Test acc (PrivateTest)',
                    'Thời gian (phút)'])
        for r in runs:
            a, b = r['args'], r['best']
            h = r['hist'][b['epoch'] - 1]
            w.writerow([r['name'], a['arch'], r['params'], 'có' if a['aug'] else 'không', a['opt'], a['lr'],
                        a['sched'], a['wd'], a['ls'], a.get('class_weights', 'none'), a['epochs'], b['epoch'],
                        f"{h['train_acc']:.4f}", f"{b['val_loss']:.4f}", f"{b['val_acc']:.4f}",
                        f"{r['test_acc_at_best_val']:.4f}", f"{r['minutes']:.1f}"])


def run_experiment(name, arch='small', aug=1, epochs=15, lr=1e-3, opt='adam', sched='none',
                   bs=64, wd=0.0, ls=0.0, data=DATA_NPZ, out=RUNS_DIR, seed=0, device='cpu',
                   class_weights='none'):
    seed_everything(seed)
    dev = get_device(device)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits(data)
    weight_tensor = None
    if class_weights not in (None, 'none'):
        counts = torch.bincount(ytr, minlength=7).float().clamp_min(1)
        if class_weights == 'sqrt':
            weight_tensor = (len(ytr) / (7 * counts)).sqrt()
        elif class_weights in ('inverse', 'balanced'):
            weight_tensor = len(ytr) / (7 * counts)
        else:
            raise ValueError(f'Unknown class_weights: {class_weights}')
        weight_tensor = weight_tensor / weight_tensor.mean()
    model = build_model(arch).to(dev)
    nparam = sum(q.numel() for q in model.parameters())
    opt_obj = make_optimizer(model, opt, lr, wd)
    steps = epochs * ((len(Xtr) + bs - 1) // bs)
    sched_obj = make_scheduler(opt_obj, sched, lr, steps)
    os.makedirs(out, exist_ok=True)
    tag = f'{out}/{name}'
    t0 = time.time()
    hist, best = fit(model, opt_obj, sched_obj, Xtr, ytr, Xva, yva, epochs, bs, bool(aug), ls, tag,
                     device=dev, class_weights=weight_tensor, t0=t0)
    load_checkpoint(model, tag + '.pt', device=dev)
    logits = predict(model, Xte, device=dev)
    args = dict(name=name, arch=arch, aug=aug, epochs=epochs, lr=lr, opt=opt, sched=sched,
                bs=bs, wd=wd, ls=ls, data=data, out=out, seed=seed, device=str(dev),
                class_weights=class_weights)
    res = dict(name=name, args=args, params=nparam, best=best, hist=hist,
               test_acc_at_best_val=(logits.argmax(1) == yte).float().mean().item(),
               minutes=(time.time() - t0) / 60)
    json.dump(res, open(tag + '.json', 'w'), indent=1)
    print('DONE', name, 'params', nparam, 'best', best, 'test', res['test_acc_at_best_val'], flush=True)
    return res
