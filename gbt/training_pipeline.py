from dataclasses import dataclass, field
from glob import glob
from os import path
from typing import Union, Callable, List
import numpy as np
import pandas as pd
from .params_preset import ParamsPreset, get_preset_params
from .dataset_preprocessor import DataPreprocessor
from .feature_transformer import FeatureTransformer
from .lightgbm_model import LightGBMModel
from .metrics import BaseMetricCalculator
from .model import GBTModel


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
    verbose: bool = True

    metrics_calculator = BaseMetricCalculator()
    model: LightGBMModel = field(init=False, default=None)
    feature_transformer: FeatureTransformer = field(init=False, default=None)

    def fit(self, dataset_builder):
        self._raise_error_if_data_not_compatible(dataset_builder)
        self.dataset_builder = dataset_builder
        df = self._load_training_dataframe(dataset_builder.training_dataset())
        feature_transformer = self._create_feature_transformer()
        train_ds, val_ds = self._preprocess_and_split_data(df, feature_transformer)
        parameters = self._get_parameters()
        if self.verbose:
            print("Parameters:")
            print(parameters)
        model = LightGBMModel(parameters=parameters, rounds=self.num_boost_round)
        model.train(train_ds, val_ds)
        self.model = model
        self.feature_transformer = feature_transformer
        if self.log_dir:
            model.booster.save_model(path.join(self.log_dir, "lgb_classifier.txt"))
            feature_transformer.save()
        # Eval
        predictions_on_train = model.predict(train_ds.data)
        predictions = model.booster.predict(val_ds.data)

        self._print_feature_importances(model)

        train_labels = train_ds.label
        val_labels = val_ds.label
        # Convert to numpy arrays if needed
        if hasattr(val_labels, "to_numpy"):
            val_labels = val_labels.to_numpy()
            train_labels = train_labels.to_numpy()
        elif hasattr(val_labels, "values"):
            val_labels = val_labels.values
            train_labels = train_labels.values

        if self.params_preset in (
            ParamsPreset.BINARY_CLASSIFICATION,
            ParamsPreset.MULTICLASS_CLASSIFICATION,
        ):
            self.metrics_calculator.task = "classification"
        else:
            self.metrics_calculator.task = "regression"

        if self.verbose:
            print("Training metrics:")
            self.metrics_calculator.run(train_labels, predictions_on_train)
            self.metrics_calculator.print()
            print("\nValidation metrics:")
            self.metrics_calculator.run(val_labels, predictions)
            self.metrics_calculator.print()
        else:
            self.metrics_calculator.run(val_labels, predictions)

        if hasattr(dataset_builder, "testing_dataset"):
            self._evaluate_on_test_data(
                model, feature_transformer, dataset_builder.testing_dataset()
            )

    def predict(self, df: Union[str, pd.DataFrame], **kwargs):
        """Predict using the trained model on new data.

        Args:
            df: Raw features as a :class:`pandas.DataFrame` or path to a CSV file.

        Returns:
            The model predictions.
        """
        if self.model is None or self.feature_transformer is None:
            raise ValueError("Model is not trained. Call `fit` first.")
        # Use GBTModel for prediction
        gbt_model = self.create_model()
        return gbt_model.predict(df, **kwargs)
    
    def create_model(self) -> GBTModel:
        """Create a GBTModel for inference.
        
        Returns:
            GBTModel instance containing booster and feature transformer
        """
        if self.model is None or self.feature_transformer is None:
            raise ValueError("Model is not trained. Call `fit` first.")
        return GBTModel(self.model.booster, self.feature_transformer)

    @classmethod
    def load(cls, model_dir: str) -> "TrainingPipeline":
        """Load a trained pipeline from saved artifacts.

        Args:
            model_dir: Directory containing ``lgb_classifier.txt`` and
                ``feature_transformer.json``.

        Returns:
            A :class:`TrainingPipeline` ready for :meth:`predict`.
        """
        # Load using GBTModel, then create pipeline wrapper for backward compatibility
        gbt_model = GBTModel.load(model_dir)
        ft = gbt_model.feature_transformer
        
        # Create a pipeline with the loaded model
        pipeline = cls(
            categorical_feature_columns=ft.categorical_features,
            numerical_feature_columns=ft.numerical_features,
            label_column=ft.target,
            log_dir=model_dir,
        )
        
        # Set the loaded components
        pipeline.model = LightGBMModel()
        pipeline.model.booster = gbt_model.booster
        pipeline.feature_transformer = ft
        
        return pipeline

    def _raise_error_if_data_not_compatible(self, dataset_builder):
        if not self.data_is_compatible(dataset_builder):
            raise ValueError(
                "The dataset is not compatible with this training pipeline."
            )

    def data_is_compatible(self, dataset_builder):
        return True

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
            if total_imp > 0:
                print(f, i / total_imp)
            else:
                print(f, 0)

    def _evaluate_on_test_data(self, model, feature_transformer, df_test):
        if df_test is not None:
            test_features = feature_transformer.transform(
                df_test, include_target_column=True
            )
            test_labels = test_features[self.label_column]
            test_features = test_features.drop(columns=[self.label_column])
            if self.verbose:
                print("test_features shape:", test_features.shape)
            test_pred = model.predict(test_features)
            if self.verbose:
                print("")
                print("On hold-out test set: ----------------------------------")
                self.metrics_calculator.run(test_labels, test_pred)
                self.metrics_calculator.print()
                print("")

    def _get_parameters(self):
        params = get_preset_params(self.params_preset)
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        # Override preset params with user params.
        if self.params_override is not None:
            if self.verbose:
                print(f"Overriding preset params with user params: {self.params_override}.")
            for k, v in self.params_override.items():
                params[k] = v
        return params

    @staticmethod
    def _load_training_dataframe(df: Union[str, pd.DataFrame]):
        if isinstance(df, str):
            filepath = df
            if path.isdir(filepath):
                dfs = []
                for fn in glob(path.join(filepath, "*.csv.*")):
                    dfs.append(pd.read_csv(fn))
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


