"""Simplified public API for training models."""
from dataclasses import dataclass
from typing import Any, Optional, List
import pandas as pd

from .training_pipeline import TrainingPipeline
from .params_preset import ParamsPreset
from .metrics import BaseMetricCalculator


@dataclass
class _DatasetBuilder:
    train_df: Any
    test_df: Any = None

    def training_dataset(self):
        return self.train_df

    def testing_dataset(self):
        return self.test_df


def train(
    df: Any,
    df_test: Any = None,
    model_lib: str = "l2",
    label_column: Optional[str] = None,
    categorical_feature_columns: Optional[List[str]] = None,
    numerical_feature_columns: Optional[List[str]] = None,
    preprocess_fn=None,
    sort_by_columns=None,
    add_categorical_stats: bool = False,
    pretrain_size: float = 0,
    val_size: float = 0.1,
    log_dir: Optional[str] = None,
    metrics_calculator: Optional[BaseMetricCalculator] = None,
    params_override: Optional[dict] = None,
    early_stopping_rounds: Optional[int] = None,
    num_boost_round: int = 30,
):
    """Train a model using :class:`TrainingPipeline`.

    Parameters mirror those of :class:`TrainingPipeline.fit` for convenience.
    """
    pipeline = TrainingPipeline(
        categorical_feature_columns=categorical_feature_columns or [],
        numerical_feature_columns=numerical_feature_columns or [],
        label_column=label_column,
        log_dir=log_dir,
        num_boost_round=num_boost_round,
        params_preset=ParamsPreset(model_lib),
        params_override=params_override,
        sort_by_columns=sort_by_columns,
        pretrain_size=pretrain_size,
        val_size=val_size,
        add_categorical_stats=add_categorical_stats,
        preprocess_fn=preprocess_fn,
        early_stopping_rounds=early_stopping_rounds,
    )
    if metrics_calculator is not None:
        pipeline.metrics_calculator = metrics_calculator
    dataset_builder = _DatasetBuilder(df, df_test)
    pipeline.fit(dataset_builder)
    return pipeline

