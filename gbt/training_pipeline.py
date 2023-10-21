from dataclasses import dataclass
from glob import glob
from os import path
from typing import Union, Callable, List
import numpy as np
import pandas as pd
from .params_preset import ParamsPreset, get_preset_params
from .dataset_preprocessor import DataPreprocessor
from .feature_transformer import FeatureTransformer
from .lightgbm_model import LightGBMModel
from .metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    BaseMetricCalculator,
)


@dataclass
class TrainingPipeline:
    """A reusable training pipeline for gradient boosted trees."""

    categorical_feature_columns: List[str]
    numerical_feature_columns: List[str]
    label_column: str
    log_dir: str = None
    num_boost_round: int = 30
    params_preset: ParamsPreset = None
    params_override: dict = None
    sort_by_columns: List[str] = None
    pretrain_size: float = None
    val_size: float = 0.2
    add_categorical_stats: bool = False
    preprocess_fn: Callable = None
    early_stopping_rounds: int = None

    metrics_calculator = BaseMetricCalculator()

    def fit(self, dataset_builder):
        self._raise_error_if_data_not_compatible(dataset_builder)
        self.dataset_builder = dataset_builder
        df = self._load_training_dataframe(dataset_builder.training_dataset())
        feature_transformer = self._create_feature_transformer()
        train_ds, val_ds = self._preprocess_and_split_data(df, feature_transformer)
        parameters = self._get_parameters()
        print("Parameters:")
        print(parameters)
        model = LightGBMModel(parameters=parameters, rounds=self.num_boost_round)
        model.train(train_ds, val_ds)
        # Eval
        predictions_on_train = model.predict(train_ds.data)
        predictions = model.booster.predict(val_ds.data)

        self._print_feature_importances(model)

        train_labels = train_ds.label
        val_labels = val_ds.label
        if hasattr(val_labels, "to_numpy"):
            val_labels = val_labels.to_numpy()
            train_labels = train_labels.to_numpy()
        elif hasattr(val_labels, "values"):
            val_labels = val_labels.values
            train_labels = train_labels.values

        self._print_regression_metrics(
            predictions_on_train, predictions, train_labels, val_labels
        )
        self._print_classification_metrics(
            predictions_on_train, predictions, train_labels, val_labels
        )
        self._print_regression_metrics(
            predictions_on_train, predictions, train_labels, val_labels
        )
        if hasattr(dataset_builder, "testing_dataset"):
            self._evaluate_on_test_data(
                model, feature_transformer, dataset_builder.testing_dataset()
            )

    def _raise_error_if_data_not_compatible(self, dataset_builder):
        if not self.data_is_compatible(dataset_builder):
            raise ValueError(
                "The dataset is not compatible with this training pipeline."
            )

    def data_is_compatible(self, dataset_builder):
        return True

    def _print_regression_metrics(
        self, predictions_on_train, predictions, train_labels, val_labels
    ):
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

        try:
            print("Top level metrics ********************")
            print(
                "mean abs err:", mean_abs_err, ", on training set:", mean_abs_err_train
            )
            mape = mean_absolute_percentage_error(val_labels, predictions)
            mape_train = mean_absolute_percentage_error(
                train_labels, predictions_on_train
            )
            print("MAPE:", mape, ", on training set:", mape_train)
            r2 = r2_score(val_labels, predictions)
            r2_train = r2_score(train_labels, predictions_on_train)
            print("R2:", r2, ", on training set:", r2_train)
            print("****************************************************\n")
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
            ds = self.dataset_builder
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

    def _print_feature_importances(self, model):
        # Print out feature importances.
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

    def _print_classification_metrics(
        self, predictions_on_train, predictions, train_labels, val_labels
    ):
        if self.params_preset == ParamsPreset.BINARY_CLASSIFICATION:
            # print("Accuracy:", accuracy_score(train_labels, predictions_on_train))
            predicted_labels_on_train = predictions_on_train > 0.5
            predicted_labels_on_val = predictions > 0.5
            print("Binary ---------------")
            print(
                "Train Accuracy:",
                accuracy_score(train_labels, predicted_labels_on_train),
            )
            print("Train AUC:", roc_auc_score(train_labels, predictions_on_train))
            print("Val Accuracy:", accuracy_score(val_labels, predicted_labels_on_val))
            print("Val AUC:", roc_auc_score(val_labels, predictions))
        if self.params_preset == ParamsPreset.MULTICLASS_CLASSIFICATION:
            predicted_labels_on_train = np.argmax(predictions_on_train, axis=1)
            print(predicted_labels_on_train)
            print(
                "Train Accuracy:",
                accuracy_score(train_labels, predicted_labels_on_train),
            )
            predicted_labels_on_val = np.argmax(predictions, axis=1)
            print("Val Accuracy:", accuracy_score(val_labels, predicted_labels_on_val))

    def _evaluate_on_test_data(self, model, feature_transformer, df_test):
        if df_test is not None:
            test_features = feature_transformer.transform(
                df_test, include_target_column=True
            )
            test_labels = test_features[self.label_column]
            test_features = test_features.drop(columns=[self.label_column])
            print("test_features shape:", test_features.shape)
            test_pred = model.predict(test_features)
            print("")
            print("On hold-out test set: ----------------------------------")
            self.metrics_calculator.run(test_labels, test_pred)
            self.metrics_calculator.print()
            print("")
            print(
                "Baseline metrics for hold-out test set: ----------------------------------"
            )

    def _get_parameters(self):
        params = get_preset_params(self.params_preset)
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        # Override preset params with user params.
        if self.params_override is not None:
            print(f"Overriding preset params with user params: {self.params_override}.")
            for k, v in self.params_override.items():
                params[k] = v
        return params

    def _load_training_dataframe(self, df: Union[str, pd.DataFrame]):
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
        elif isinstance(df, pd.DataFrame):
            pass
        else:
            raise TypeError(f"Unsupported type: {type(df)}")
        return df

    def _create_feature_transformer(self):
        feature_transformer = FeatureTransformer(
            categorical_features=self.categorical_feature_columns,
            numerical_features=self.numerical_feature_columns,
            target=self.label_column,
            output_dir=self.log_dir,
            add_categorical_stats=self.add_categorical_stats,
            preprocess_fn=self.preprocess_fn,
        )
        return feature_transformer

    def _preprocess_and_split_data(
        self,
        df: pd.DataFrame,
        feature_transformer: FeatureTransformer,
    ):
        # This class needs renamed to DataPreprocessor.
        ds = DataPreprocessor(
            local_dir_or_file=None,
            log_dir=self.log_dir,
            feature_transformer=feature_transformer,
            sort_by_columns=self.sort_by_columns,
            label_column=self.label_column,
        )
        ds.df = df
        ds.preprocess()
        if ds.features.shape[0] <= 10:
            import warnings

            warnings.warn(
                f"Too few samples: {ds.features.shape[0]}. Training may not converge."
            )
        train_ds, val_ds = ds.split(
            pretrain_size=self.pretrain_size, val_size=self.val_size, shuffle=False
        )
        return train_ds, val_ds


if __name__ == "__main__":

    class DataPreprocessor:
        def training_dataset(self):
            df = pd.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5, 6, 7],
                    "b": ["a", "b", "c", None, "e", "f", "g"],
                    "c": [1, 0, 1, 1, 0, 0, 1],
                    "some_other_column": [0, 0, None, None, None, 3, 3],
                }
            )
            return df

        def testing_dataset(self):
            return self.training_dataset()

    TrainingPipeline(
        params_preset="binary",  # one of mape, l2, binary, multiclass
        params_override={"num_leaves": 10},
        label_column="c",
        val_size=0.2,  # fraction of the validation split
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
    ).fit(DataPreprocessor())
