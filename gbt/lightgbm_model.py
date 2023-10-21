from glob import glob
from os import path

import lightgbm as lgb

from .feature_transformer import FeatureTransformer


class LightGBMModel:
    def __init__(self, parameters=None, rounds=None, patience=None, load_model=False):
        """
        Args
        ========================
        parameters: dict
            LightGBM parameters.
        """
        self.parameters = parameters if parameters else {}
        self.rounds = rounds or 10
        self.patience = patience or 10
        self.booster = None
        self.feature_transformer = None
        if load_model:
            assert (
                self.load_latest_model("tmp/trained_models") is not None
            ), "Cannot load model!"

    def load_model(self, experiment_dir):
        model_path = path.join(experiment_dir, "lgb_classifier.txt")
        feature_transformer_path = path.join(experiment_dir, "feature_transformer.json")
        if path.exists(model_path) and path.exists(feature_transformer_path):
            self.booster = lgb.Booster(model_file=model_path)
            self.feature_transformer = FeatureTransformer(output_dir=experiment_dir)
            return self.booster
        print("files not found")
        print("model path:", model_path)
        print("feature transformer path:", feature_transformer_path)
        return None

    def load_latest_model(self, model_dir):
        experiment_dirs = sorted(glob(path.join(model_dir, "*")))
        experiment_dirs = experiment_dirs[::-1]
        for experiment_dir in experiment_dirs:
            if self.load_model(experiment_dir):
                return self.booster
        return None

    def train(self, train_dataset, val_dataset):
        print("lgb parameters:")
        print(self.parameters)
        self.booster = lgb.train(
            params=self.parameters,
            train_set=train_dataset,
            valid_sets=val_dataset,
            num_boost_round=self.rounds,
        )

    def transform(self, X):
        return self.feature_transformer.transform(X)

    def predict(self, X, **kwargs):
        return self.booster.predict(X, **kwargs)
