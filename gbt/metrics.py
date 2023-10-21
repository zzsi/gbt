import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
)


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon, y_true))) * 100


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

class BaseMetricCalculator:
    def __init__(self, task="regression"):
        """
        task: str, 'regression' or 'classification'.
        """
        self.task = task

    def run(self, y_true, y_pred, features=None):
        if self.task == "regression":
            self.result = self.calculate_for_regression(y_true, y_pred, features)
        elif self.task == "classification":
            self.result = self.calculate_for_classification(y_true, y_pred, features)
        return self.result

    def print(self):
        for k in ["r2", "mape", "mae", "mae_log10"]:
            value = self.result[k]
            print(f"{k.upper()}: {value}")

    def calculate_for_regression(self, y_true, y_pred, features=None):
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae_log10 = mean_log10_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"r2": r2, "mape": mape, "mae": mae, "mae_log10": mae_log10}

    def calculate_for_classification(self, y_true, y_pred, features=None):
        raise NotImplementedError