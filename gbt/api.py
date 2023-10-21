"""Top level methods for boosted_tree_kungfu.
"""
from glob import glob
from os import path

import numpy as np
import pandas as pd

from .dataset_preprocessor import DataPreprocessor
from .feature_transformer import FeatureTransformer
from .lightgbm_model import LightGBMModel
from .metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_log10_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    BaseMetricCalculator,
)

csd = path.dirname(path.realpath(__file__))


def train(
    df,
    df_test=None,
    recipe="l2",
    num_classes=None,
    label_column=None,
    categorical_feature_columns=None,
    numerical_feature_columns=None,
    preprocess_fn=None,
    sort_by_columns=None,
    add_categorical_stats=False,
    pretrain_size=0,
    val_size=0.1,
    log_dir=None,
    metrics_calculator=BaseMetricCalculator(),
):
    """
    TODO: remove `recipe`. Add `model_lib` (sklearn, lgb), and common tree parameters.
    """
    if isinstance(df, str):
        filepath = df
        if path.isdir(filepath):
            dfs = []
            for fn in glob(path.join(filepath, "*.csv.*")):
                dfs.append(df.read_csv(fn))
            df = pd.concat(dfs)
        elif path.isfile(filepath):
            df = pd.read_csv(filepath)
        else:
            raise OSError(f"No such file: {filepath}")

    feature_transformer = FeatureTransformer(
        categorical_features=categorical_feature_columns,
        numerical_features=numerical_feature_columns,
        target=label_column,
        output_dir=log_dir,
        add_categorical_stats=add_categorical_stats,
        preprocess_fn=preprocess_fn,
    )

    ds = DataPreprocessor(
        local_dir_or_file=None,
        log_dir=None,  # path.join(csd, "model_output"),
        feature_transformer=feature_transformer,
        sort_by_columns=sort_by_columns,
        label_column=label_column,
    )
    ds.df = df
    ds.preprocess()
    if ds.features.shape[0] <= 10:
        import warnings

        warnings.warn(
            f"Too few samples: {ds.features.shape[0]}. Training may not converge."
        )
    train_ds, val_ds = ds.split(
        pretrain_size=pretrain_size, val_size=val_size, shuffle=False
    )

    if recipe == "mape":
        parameters = {
            "boosting_type": "gbdt",
            "metric": "mape",
            "objective": "mape",
            "learning_rate": 0.03,
            "num_leaves": 255,
            "min_data": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }
    elif recipe == "l2":
        parameters = {
            "boosting_type": "gbdt",
            "metric": "l2",
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data": 20,
            # "lambda_l1": 0.001,
            # "lambda_l2": 0.001,
            "verbosity": 1,
        }
    elif recipe == "l2_rf":
        parameters = {
            "boosting_type": "rf",
            "metric": "l2",
            "objective": "regression",
            "learning_rate": 0.03,
            "min_data": 20,
            "bagging_freq": 1,
            "bagging_fraction": 0.65,
            "feature_fraction": 0.6,
            "num_leaves": 127,
        }
    elif recipe == "binary":
        parameters = {
            "boosting_type": "gbdt",
            "metric": "binary_logloss",
            # "is_unbalance": True,
            "objective": "binary",
            # "learning_rate": 0.03,
            "min_data_in_leaf": 2,
            "num_leaves": 127,
        }
    elif recipe == "multiclass":
        parameters = {
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
    else:
        raise ValueError(
            f"Unknown recipe: {recipe}. Supported: mape, l2, l2_rf, binary, multiclass"
        )

    print(parameters)
    model = LightGBMModel(parameters=parameters, rounds=100)
    model.train(train_ds, val_ds)
    # Eval
    predictions_on_train = model.predict(train_ds.data)
    predictions = model.booster.predict(val_ds.data)

    # Print out feature importances.
    if "sk_" not in recipe:
        print("Feature importances:")
        total_imp = np.sum(model.booster.feature_importance(importance_type="gain"))
        features_and_gains = list(
            zip(
                model.booster.feature_name(),
                model.booster.feature_importance(importance_type="gain"),
            )
        )
        for f, i in sorted(features_and_gains, key=lambda x: -x[1]):
            print(f, i / total_imp)

    train_labels = train_ds.label
    val_labels = val_ds.label
    if hasattr(val_labels, "to_numpy"):
        val_labels = val_labels.to_numpy()
        train_labels = train_labels.to_numpy()
    elif hasattr(val_labels, "values"):
        val_labels = val_labels.values
        train_labels = train_labels.values

    try:
        mean_abs_err_train = mean_absolute_error(train_labels, predictions_on_train)
        mean_abs_err = mean_absolute_error(val_labels, predictions)

        def mael10(y_true, y_pred, epsilon=1):
            return np.abs(
                np.log10(np.maximum(epsilon, y_true))
                - np.log10(np.maximum(epsilon, y_pred))
            ).mean()

        mean_abs_err_log10_train = mael10(train_labels, predictions_on_train)
        mean_abs_err_log10 = mael10(val_labels, predictions)
        print(
            "MAE log10:",
            mean_abs_err_log10,
            ", on training set:",
            mean_abs_err_log10_train,
        )
    except:
        pass

    def median_log10_error(y_true, y_pred, epsilon=1):
        return np.median(
            np.abs(
                np.log10(np.maximum(epsilon, y_true))
                - np.log10(np.maximum(epsilon, y_pred))
            )
        )

    try:
        median_abs_error_log10_train = median_log10_error(
            train_labels, predictions_on_train
        )
        median_abs_error_log10 = median_log10_error(val_labels, predictions)

        print("Worst predictions:")
        worst_predictions = sorted(
            enumerate(np.abs(val_labels - predictions)), key=lambda x: -x[1]
        )[:10]
        #     print(worst_predictions)
        for i, diff in worst_predictions:
            print(ds.val_features.iloc[i].to_dict())
            try:
                print(
                    "actual:",
                    val_labels[i],
                    "predicted:",
                    predictions[i],
                    "diff:",
                    diff,
                )
            except:
                print(val_labels[:20])
                print(predictions[:20])
                raise
            print("----------------------------\n")
    except:
        pass

    try:
        # TODO: need abstraction.
        print()
        print("-------------------------------------")
        print()
        print("Price:")
        ds.val_features["SalePrice"] = val_labels
        print(
            ds.val_features.groupby("CategoryName")["SalePrice"].agg(["mean", "count"])
        )
        print("-------------------------------------")
        print("Absolute Percentage Error: slicing and dicing")
        ds.val_features["ape"] = (
            np.abs(val_labels - predictions) / np.maximum(1, val_labels) * 100
        )
        error_by_category = ds.val_features.groupby("CategoryName")["ape"].agg(
            ["mean", "count"]
        )
        print(error_by_category)
        error_by_root_type = ds.val_features.groupby("RootType")["ape"].agg(
            ["mean", "count"]
        )
        print(error_by_root_type)
    except Exception:
        pass

    if recipe == "binary":
        # print("Accuracy:", accuracy_score(train_labels, predictions_on_train))
        predicted_labels_on_train = predictions_on_train > 0.5
        predicted_labels_on_val = predictions > 0.5
        print("Binary ---------------")
        print(
            "Train Accuracy:", accuracy_score(train_labels, predicted_labels_on_train)
        )
        print("Train AUC:", roc_auc_score(train_labels, predictions_on_train))
        print("Val Accuracy:", accuracy_score(val_labels, predicted_labels_on_val))
        print("Val AUC:", roc_auc_score(val_labels, predictions))
    if recipe == "multiclass":
        predicted_labels_on_train = np.argmax(predictions_on_train, axis=1)
        print(predicted_labels_on_train)
        print(
            "Train Accuracy:", accuracy_score(train_labels, predicted_labels_on_train)
        )
        predicted_labels_on_val = np.argmax(predictions, axis=1)
        print("Val Accuracy:", accuracy_score(val_labels, predicted_labels_on_val))
        # print("AUC:", roc_auc_score(train_labels, predicted_labels_on_train))

    try:
        print("Top level metrics ********************")
        print("mean abs err:", mean_abs_err, ", on training set:", mean_abs_err_train)
        mape = mean_absolute_percentage_error(val_labels, predictions)
        mape_train = mean_absolute_percentage_error(train_labels, predictions_on_train)
        print("MAPE:", mape, ", on training set:", mape_train)
        r2 = r2_score(val_labels, predictions)
        r2_train = r2_score(train_labels, predictions_on_train)
        print("R2:", r2, ", on training set:", r2_train)

        print(
            "Median log10 error",
            median_abs_error_log10,
            ", on training set",
            median_abs_error_log10_train,
        )
        print("****************************************************\n")
    except:
        pass

    if df_test is not None:
        test_features = feature_transformer.transform(
            df_test, include_target_column=True
        )
        test_labels = test_features[label_column]
        test_features = test_features.drop(columns=[label_column])
        print("test_features shape:", test_features.shape)
        test_pred = model.predict(test_features)
        print("")
        print("On hold-out test set: ----------------------------------")
        metrics_calculator.run(test_labels, test_pred)
        metrics_calculator.print()

    return model
