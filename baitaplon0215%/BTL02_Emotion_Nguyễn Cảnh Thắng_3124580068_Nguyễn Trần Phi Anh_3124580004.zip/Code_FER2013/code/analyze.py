"""Phân tích sau huấn luyện: nhật ký, đường cong, confusion matrix, ảnh đúng/sai,
trực quan hóa filter conv cuối và các hình dữ liệu."""
import argparse
import json
import os

import numpy as np

from fer.config import ARCH, BEST_REPORT, DATA_NPZ, EXPERIMENT_LOG, FIG_DIR, RUNS_DIR
from fer.data import load_fer, to_tensor
from fer.engine import load_checkpoint, predict
from fer.experiments import best_run, load_runs, write_experiment_log
from fer.metrics import confusion, report_dict, report_text
from fer.models import FERNet
from fer.plots import (save_augmentation, save_case_grid, save_class_hist, save_compare_curves,
                       save_confusion_matrix, save_run_curves, save_samples)
from fer.viz import activation_maximization, conv_layers, feature_extractor, save_feature_maps, show_kernels


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Phân tích kết quả các thí nghiệm FER2013.')
    p.add_argument('--runs', default=RUNS_DIR)
    p.add_argument('--figures', default=FIG_DIR)
    p.add_argument('--data', default=DATA_NPZ)
    p.add_argument('--log', default=EXPERIMENT_LOG)
    p.add_argument('--report', default=BEST_REPORT)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.figures, exist_ok=True)
    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit(f'Không tìm thấy file E*.json trong {a.runs}')
    write_experiment_log(runs, a.log)

    save_compare_curves(runs, os.path.join(a.figures, 'compare_val.png'))
    for r in runs:
        save_run_curves(r, os.path.join(a.figures, f"curve_{r['name'].split('_')[0]}.png"))

    best = best_run(runs)
    print('BEST', best['name'])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer(a.data)
    model = FERNet(**ARCH[best['args']['arch']])
    load_checkpoint(model, os.path.join(a.runs, best['name'] + '.pt'))
    model.eval()
    Xt = to_tensor(Xte)
    logits = predict(model, Xt)
    pred = logits.argmax(1).numpy()
    acc = (pred == yte).mean()

    cm = confusion(yte, pred)
    save_confusion_matrix(cm, best['name'], acc, os.path.join(a.figures, 'confusion_matrix.png'))
    rep = report_dict(yte, pred)
    json.dump(dict(best=best['name'], test_acc=float(acc), report=rep, cm=cm.tolist()),
              open(a.report, 'w'), indent=1)

    rng = np.random.RandomState(0)
    wrong = np.where(pred != yte)[0]
    right = np.where(pred == yte)[0]
    save_case_grid(Xte, yte, pred, rng.choice(wrong, 32, replace=False),
                   os.path.join(a.figures, 'wrong_cases.png'), 'Dự đoán sai (Predict (True))')
    save_case_grid(Xte, yte, pred, rng.choice(right, 32, replace=False),
                   os.path.join(a.figures, 'true_cases.png'), 'Dự đoán đúng (Predict (True))')

    convs = conv_layers(model)
    first, last = convs[0], convs[-1]
    show_kernels(first, os.path.join(a.figures, 'filters_first_conv.png'), 'Kernel 3x3 của conv đầu tiên')
    show_kernels(last, os.path.join(a.figures, 'filters_last_conv_weights.png'),
                 f'Kernel 3x3 của conv cuối (trung bình theo kênh vào), {last.out_channels} filter, hiện 32 đầu')
    sub = feature_extractor(model)
    activation_maximization(sub, os.path.join(a.figures, 'filters_last_conv_actmax.png'))
    i0 = int(right[np.where(yte[right] == 3)[0][0]])
    save_feature_maps(sub, Xt, Xte, i0, os.path.join(a.figures, 'feature_maps_last_conv.png'))

    save_class_hist(ytr, yva, yte, os.path.join(a.figures, 'class_hist.png'))
    save_samples(Xtr, ytr, os.path.join(a.figures, 'samples.png'))
    save_augmentation(Xtr, os.path.join(a.figures, 'augmentation.png'))
    print(report_text(yte, pred))


if __name__ == '__main__':
    main()
