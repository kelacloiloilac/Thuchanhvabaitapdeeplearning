"""Chỉ số đánh giá: confusion matrix và classification report."""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from .config import CLASSES


def confusion(y_true, y_pred):
    return sk_confusion_matrix(y_true, y_pred)


def report_dict(y_true, y_pred, classes=CLASSES):
    return classification_report(y_true, y_pred, target_names=classes, output_dict=True)


def report_text(y_true, y_pred, classes=CLASSES):
    return classification_report(y_true, y_pred, target_names=classes)
