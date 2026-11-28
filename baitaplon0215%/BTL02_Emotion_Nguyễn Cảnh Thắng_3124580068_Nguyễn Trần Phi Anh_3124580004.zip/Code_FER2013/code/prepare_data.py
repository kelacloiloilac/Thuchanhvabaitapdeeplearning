"""Chuyển CSV FER2013 (emotion,pixels,Usage) sang fer2013.npz cho train.py.

Ví dụ: python prepare_data.py --csv ../icml_face_data.csv --out ../fer2013.npz
"""
import argparse
import csv
import os

import numpy as np

from fer.config import BASE_DIR, DATA_NPZ

USAGES = ('Training', 'PublicTest', 'PrivateTest')
EXPECTED = {'Training': 28709, 'PublicTest': 3589, 'PrivateTest': 3589}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Tạo fer2013.npz từ CSV FER2013.')
    p.add_argument('--csv', default=os.path.join(BASE_DIR, 'icml_face_data.csv'))
    p.add_argument('--out', default=DATA_NPZ)
    return p.parse_args(argv)


def read_csv(path):
    X, y, usage = [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        missing = {'emotion', 'pixels', 'usage'} - set(fields)
        if missing:
            raise ValueError(f'Thiếu cột {sorted(missing)} trong {path}; có: {reader.fieldnames}')
        for row in reader:
            X.append(np.array(row[fields['pixels']].split(), dtype=np.uint8).reshape(48, 48))
            y.append(int(row[fields['emotion']]))
            usage.append(row[fields['usage']].strip())
    return np.stack(X), np.array(y, dtype=np.int64), np.array(usage)


def main(argv=None):
    a = parse_args(argv)
    X, y, usage = read_csv(a.csv)
    unknown = sorted(set(usage) - set(USAGES))
    if unknown:
        raise ValueError(f'Giá trị Usage lạ: {unknown}')
    counts = {u: int((usage == u).sum()) for u in USAGES}
    print('Số ảnh:', counts)
    for u, n in EXPECTED.items():
        if counts[u] != n:
            print(f'Cảnh báo: {u} có {counts[u]} ảnh, kỳ vọng {n}')
    np.savez_compressed(a.out, X=X, y=y, usage=usage)
    print('Đã lưu', a.out)


if __name__ == '__main__':
    main()
