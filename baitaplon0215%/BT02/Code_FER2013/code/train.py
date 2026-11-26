"""Huấn luyện 1 thí nghiệm. Ví dụ: python train.py --name E2 --arch small --aug 1 --epochs 15"""
import argparse, json, time, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fer_lib import *
p = argparse.ArgumentParser()
p.add_argument('--name', required=True); p.add_argument('--arch', default='small')
p.add_argument('--aug', type=int, default=1); p.add_argument('--epochs', type=int, default=15)
p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--opt', default='adam')  # adam | sgd
p.add_argument('--sched', default='none')  # none | cosine
p.add_argument('--bs', type=int, default=64); p.add_argument('--wd', type=float, default=0.0)
p.add_argument('--ls', type=float, default=0.0, help='label smoothing')
p.add_argument('--data', default='../../fer2013.npz'); p.add_argument('--out', default='../runs')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(max(1, os.cpu_count() or 1))
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer(a.data)
Xtr, Xva, Xte = map(to_tensor, (Xtr, Xva, Xte)); ytr, yva, yte = map(torch.from_numpy, (ytr, yva, yte))
model = FERNet(**ARCH[a.arch]); nparam = sum(q.numel() for q in model.parameters())
opt = (torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.opt == 'adam'
       else torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd))
steps = a.epochs * ((len(Xtr) + a.bs - 1) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps) if a.sched == 'cosine' else None
os.makedirs(a.out, exist_ok=True); tag = f'{a.out}/{a.name}'
hist, best = [], dict(val_acc=0, epoch=-1); t0 = time.time()
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(Xtr)); tl = tc = 0
    for i in range(0, len(Xtr), a.bs):
        idx = perm[i:i + a.bs]; x, y = Xtr[idx], ytr[idx]
        if a.aug: x = augment(x)
        opt.zero_grad(); out = model(x); loss = F.cross_entropy(out, y, label_smoothing=a.ls)
        loss.backward(); opt.step()
        if sched: sched.step()
        tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item()
    lv = predict(model, Xva); vl = F.cross_entropy(lv, yva).item(); va = (lv.argmax(1) == yva).float().mean().item()
    r = dict(epoch=ep + 1, train_loss=tl / len(Xtr), train_acc=tc / len(Xtr), val_loss=vl, val_acc=va,
             lr=opt.param_groups[0]['lr'], elapsed_min=(time.time() - t0) / 60)
    hist.append(r); print(json.dumps(r), flush=True)
    if va > best['val_acc']:
        best = dict(val_acc=va, val_loss=vl, epoch=ep + 1); torch.save(model.state_dict(), tag + '.pt')
model.load_state_dict(torch.load(tag + '.pt')); lt = predict(model, Xte)
res = dict(name=a.name, args=vars(a), params=nparam, best=best, hist=hist,
           test_acc_at_best_val=(lt.argmax(1) == yte).float().mean().item(), minutes=(time.time() - t0) / 60)
json.dump(res, open(tag + '.json', 'w'), indent=1)
print('DONE', a.name, 'params', nparam, 'best', best, 'test', res['test_acc_at_best_val'], flush=True)
