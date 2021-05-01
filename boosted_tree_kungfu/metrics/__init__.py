import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (
        np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon, y_true))) * 100
    )


def median_log10_error(y_true, y_pred, epsilon=1):
    return np.median(
        np.abs(
            np.log10(np.maximum(epsilon, y_true))
            - np.log10(np.maximum(epsilon, y_pred))
        )
    )


def mean_log10_error(y_true, y_pred, epsilon=1):
    return np.mean(
        np.abs(
            np.log10(np.maximum(epsilon, y_true))
            - np.log10(np.maximum(epsilon, y_pred))
        )
    )
