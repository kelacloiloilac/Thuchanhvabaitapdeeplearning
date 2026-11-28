"""Huấn luyện 1 thí nghiệm FER2013 bằng PyTorch.

Ví dụ: python train.py --name E2 --arch small --aug 1 --epochs 15
"""
import argparse

from fer.config import DATA_NPZ, RUNS_DIR
from fer.experiments import run_experiment


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Huấn luyện một cấu hình CNN cho FER2013.')
    p.add_argument('--name', required=True)
    p.add_argument('--arch', default='small')
    p.add_argument('--aug', type=int, default=1)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--opt', default='adam')  # adam | sgd
    p.add_argument('--sched', default='none')  # none | cosine
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--ls', type=float, default=0.0, help='label smoothing')
    p.add_argument('--data', default=DATA_NPZ)
    p.add_argument('--out', default=RUNS_DIR)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', help='cpu | cuda | auto')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run_experiment(name=a.name, arch=a.arch, aug=a.aug, epochs=a.epochs, lr=a.lr, opt=a.opt,
                   sched=a.sched, bs=a.bs, wd=a.wd, ls=a.ls, data=a.data, out=a.out, seed=a.seed,
                   device=a.device)


if __name__ == '__main__':
    main()
