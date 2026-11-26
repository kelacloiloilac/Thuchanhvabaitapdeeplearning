"""Phân tích sau huấn luyện: bảng nhật ký thí nghiệm, đường cong, confusion matrix, ảnh đúng/sai, trực quan hóa filter conv cuối."""
import json, glob, os, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from fer_lib import *
RUNS, FIG = '../runs', '../figures'; os.makedirs(FIG, exist_ok=True)
runs = sorted([json.load(open(f)) for f in glob.glob(f'{RUNS}/E*.json')], key=lambda r: r['name'])
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer('../../fer2013.npz')

# ---- 1. Nhật ký thí nghiệm (CSV) ----
import csv
with open('../experiment_log.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['Thí nghiệm','Kiến trúc','Số tham số','Augmentation','Optimizer','LR','Scheduler','Weight decay','Label smoothing','Epochs','Best epoch','Train acc @best','Val loss @best','Val acc (PublicTest)','Test acc (PrivateTest)','Thời gian (phút)'])
    for r in runs:
        a, b = r['args'], r['best']; h = r['hist'][b['epoch'] - 1]
        w.writerow([r['name'], a['arch'], r['params'], 'có' if a['aug'] else 'không', a['opt'], a['lr'], a['sched'], a['wd'], a['ls'], a['epochs'], b['epoch'],
                    f"{h['train_acc']:.4f}", f"{b['val_loss']:.4f}", f"{b['val_acc']:.4f}", f"{r['test_acc_at_best_val']:.4f}", f"{r['minutes']:.1f}"])
# ---- 2. Đường cong từng thí nghiệm + so sánh ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for r in runs:
    e = [h['epoch'] for h in r['hist']]
    ax[0].plot(e, [h['val_acc'] for h in r['hist']], label=r['name'].split('_')[0] + ' ' + '_'.join(r['name'].split('_')[1:]))
    ax[1].plot(e, [h['val_loss'] for h in r['hist']])
ax[0].set_title('Validation accuracy (PublicTest)'); ax[1].set_title('Validation loss'); [x.set_xlabel('epoch') for x in ax]; ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{FIG}/compare_val.png', dpi=130); plt.close()
for r in runs:
    h = r['hist']; e = [x['epoch'] for x in h]; fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(e, [x['train_acc'] for x in h], label='train'); ax[0].plot(e, [x['val_acc'] for x in h], label='val'); ax[0].set_title('Accuracy'); ax[0].legend()
    ax[1].plot(e, [x['train_loss'] for x in h], label='train'); ax[1].plot(e, [x['val_loss'] for x in h], label='val'); ax[1].set_title('Loss'); ax[1].legend()
    [x.grid(alpha=.3) for x in ax]; [x.set_xlabel('epoch') for x in ax]; fig.suptitle(r['name']); plt.tight_layout(); plt.savefig(f"{FIG}/curve_{r['name'].split('_')[0]}.png", dpi=120); plt.close()

# ---- 3. Mô hình tốt nhất (theo val acc) ----
best = max(runs, key=lambda r: r['best']['val_acc']); print('BEST', best['name'])
model = FERNet(**ARCH[best['args']['arch']]); model.load_state_dict(torch.load(f"{RUNS}/{best['name']}.pt")); model.eval()
Xt = to_tensor(Xte); logits = predict(model, Xt); pred = logits.argmax(1).numpy(); acc = (pred == yte).mean()
cm = confusion_matrix(yte, pred); cmn = cm / cm.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6.5, 5.6)); im = ax.imshow(cmn, cmap='Blues'); ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(CLASSES, rotation=45); ax.set_yticklabels(CLASSES); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(7):
    for j in range(7): ax.text(j, i, f'{cmn[i,j]:.0f}%', ha='center', va='center', color='white' if cmn[i, j] > 50 else 'black', fontsize=9)
ax.set_title(f"Confusion matrix – {best['name']} (PrivateTest acc {acc*100:.2f}%)", fontsize=10); plt.colorbar(im); plt.tight_layout(); plt.savefig(f'{FIG}/confusion_matrix.png', dpi=130); plt.close()
rep = classification_report(yte, pred, target_names=CLASSES, output_dict=True); json.dump(dict(best=best['name'], test_acc=float(acc), report=rep, cm=cm.tolist()), open('../best_report.json', 'w'), indent=1)
def grid(idx, fname, title):
    fig, axs = plt.subplots(4, 8, figsize=(12, 6.8))
    for k, ax in enumerate(axs.ravel()):
        ax.axis('off')
        if k < len(idx): i = idx[k]; ax.imshow(Xte[i], cmap='gray'); ax.set_title(f'{CLASSES[pred[i]]} ({CLASSES[yte[i]]})', fontsize=8, color='green' if pred[i] == yte[i] else 'red')
    fig.suptitle(title); plt.tight_layout(); plt.savefig(f'{FIG}/{fname}', dpi=110); plt.close()
rng = np.random.RandomState(0); wrong = np.where(pred != yte)[0]; right = np.where(pred == yte)[0]
grid(rng.choice(wrong, 32, replace=False), 'wrong_cases.png', 'Dự đoán sai (Predict (True))'); grid(rng.choice(right, 32, replace=False), 'true_cases.png', 'Dự đoán đúng (Predict (True))')

# ---- 4. Trực quan hóa filter lớp conv cuối ----
convs = [m for m in model.features if isinstance(m, nn.Conv2d)]; last = convs[-1]; first = convs[0]
def show_kernels(conv, fname, title, nmax=32):
    W = conv.weight.detach().numpy()  # (out,in,3,3)
    if W.shape[1] == 1:
        fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < W.shape[0]: ax.imshow(W[k, 0], cmap='bwr', vmin=-abs(W).max(), vmax=abs(W).max())
    else:  # trung bình theo kênh vào -> mỗi filter còn 3x3
        M = np.abs(W).mean(1) * 0 + W.mean(1); fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < min(nmax, M.shape[0]): ax.imshow(M[k], cmap='bwr', vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    fig.suptitle(title); plt.tight_layout(); plt.savefig(f'{FIG}/{fname}', dpi=120); plt.close()
show_kernels(first, 'filters_first_conv.png', 'Kernel 3x3 của conv đầu tiên')
show_kernels(last, 'filters_last_conv_weights.png', f'Kernel 3x3 của conv cuối (trung bình theo kênh vào), {last.out_channels} filter, hiện 32 đầu')
# Activation maximization: ảnh làm kích hoạt mạnh nhất từng filter conv cuối
feat_to_last = nn.Sequential(*list(model.features)[:list(model.features).index(last) + 1])  # (đến Conv cuối; BN/ReLU sau đó tính riêng)
idx_last = list(model.features).index(last); sub = nn.Sequential(*list(model.features)[:idx_last + 3 if idx_last + 2 < len(model.features) and isinstance(model.features[idx_last+1], nn.BatchNorm2d) else idx_last + 2])
for p_ in sub.parameters(): p_.requires_grad_(False)
sub.eval(); torch.manual_seed(0); nf = 32; x = (torch.randn(nf, 1, 48, 48) * 0.1).requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
for it in range(150):
    opt.zero_grad(); a = sub(x); loss = -a[torch.arange(nf), torch.arange(nf)].mean() + 1e-3 * (x ** 2).mean(); loss.backward(); opt.step()
fig, axs = plt.subplots(4, 8, figsize=(11, 6))
for k, ax in enumerate(axs.ravel()): ax.imshow(x[k, 0].detach().numpy(), cmap='gray'); ax.axis('off'); ax.set_title(f'f{k}', fontsize=7)
fig.suptitle('Activation maximization: mẫu đầu vào kích hoạt mạnh nhất 32 filter đầu của conv cuối'); plt.tight_layout(); plt.savefig(f'{FIG}/filters_last_conv_actmax.png', dpi=120); plt.close()
# Feature maps của 1 ảnh mẫu qua conv cuối
i0 = int(right[np.where(yte[right] == 3)[0][0]]); fm = sub(Xt[i0:i0 + 1])[0].detach().numpy()
fig, axs = plt.subplots(4, 9, figsize=(12, 5.6)); axs[0, 0].imshow(Xte[i0], cmap='gray'); axs[0, 0].set_title('Input (Happy)', fontsize=7)
for ax in axs.ravel(): ax.axis('off')
for k in range(31): axs.ravel()[k + 1].imshow(fm[k], cmap='viridis'); axs.ravel()[k + 1].set_title(f'f{k}', fontsize=6)
fig.suptitle('Feature map của conv cuối (6x6) cho một ảnh Happy'); plt.tight_layout(); plt.savefig(f'{FIG}/feature_maps_last_conv.png', dpi=120); plt.close()
# Phân bố dữ liệu + ảnh mẫu
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a_, (n, y) in zip(ax, [('Training', ytr), ('PublicTest', yva), ('PrivateTest', yte)]): a_.bar(CLASSES, np.bincount(y, minlength=7)); a_.set_title(f'{n} ({len(y)})'); a_.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(f'{FIG}/class_hist.png', dpi=120); plt.close()
fig, axs = plt.subplots(2, 7, figsize=(12, 4))
for c in range(7):
    ids = np.where(ytr == c)[0][:2]
    for r_, i in enumerate(ids): axs[r_, c].imshow(Xtr[i], cmap='gray'); axs[r_, c].axis('off'); axs[r_, c].set_title(CLASSES[c], fontsize=8) if r_ == 0 else None
plt.tight_layout(); plt.savefig(f'{FIG}/samples.png', dpi=120); plt.close()
from fer_lib import augment
xs = to_tensor(Xtr[:8]); fig, axs = plt.subplots(3, 8, figsize=(12, 4.8)); torch.manual_seed(1)
for r_ in range(3):
    xa = xs if r_ == 0 else augment(xs)
    for c in range(8): axs[r_, c].imshow(xa[c, 0], cmap='gray'); axs[r_, c].axis('off')
axs[0, 0].set_title('gốc', fontsize=8); axs[1, 0].set_title('augment', fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/augmentation.png', dpi=110); plt.close()
print(classification_report(yte, pred, target_names=CLASSES))
