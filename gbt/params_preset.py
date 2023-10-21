import enum


class ParamsPreset(str, enum.Enum):
    BINARY_CLASSIFICATION = "binary"
    MULTICLASS_CLASSIFICATION = "multiclass"
    REGRESSION_L2 = "l2"
    REGRESSION_MAPE = "mape"


def get_preset_params(preset: ParamsPreset, num_classes: int = None) -> dict:
    if preset == ParamsPreset.BINARY_CLASSIFICATION:
        return {
            "boosting_type": "gbdt",
            "metric": "binary_logloss",
            "objective": "binary",
            "min_data_in_leaf": 2,
            "num_leaves": 127,
            # "is_unbalance": True,
            # "learning_rate": 0.03,
        }
    elif preset == ParamsPreset.MULTICLASS_CLASSIFICATION:
        return {
            "boosting_type": "gbdt",
            "metric": "multi_logloss",
            "objective": "multiclassova",
            "is_unbalance": True,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data": 20,
            "lambda_l1": 0.001,
            "lambda_l2": 0.001,
            "num_class": num_classes,
        }
    elif preset == ParamsPreset.REGRESSION_L2:
        return {
            "boosting_type": "gbdt",
            "metric": "l2",
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data": 20,
            "verbosity": 1,
        }
    elif preset == ParamsPreset.REGRESSION_MAPE:
        return {
            "boosting_type": "gbdt",
            "metric": "mape",
            "objective": "mape",
            "learning_rate": 0.03,
            "num_leaves": 255,
            "min_data": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }
    else:
        return None
