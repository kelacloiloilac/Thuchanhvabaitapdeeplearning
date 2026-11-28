"""Cấu hình dùng chung: nhãn, kiến trúc và đường dẫn mặc định."""
import os

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

ARCH = {
    'small': dict(channels=(16, 32, 64), reps=(1, 1, 1)),
    'medium': dict(channels=(32, 64, 128), reps=(1, 2, 2)),
    'small_nobn': dict(channels=(16, 32, 64), reps=(1, 1, 1), bn=False),
}

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(PACKAGE_DIR)
EXPERIMENT_DIR = os.path.dirname(CODE_DIR)
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)

DATA_NPZ = os.path.join(BASE_DIR, 'fer2013.npz')
RUNS_DIR = os.path.join(EXPERIMENT_DIR, 'runs')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')
EXPERIMENT_LOG = os.path.join(EXPERIMENT_DIR, 'experiment_log.csv')
BEST_REPORT = os.path.join(EXPERIMENT_DIR, 'best_report.json')
