from glob import glob
from os import path

import lightgbm as lgb


class LightGBMModel:
    def __init__(self, parameters=None, rounds=None):
        """
        Args
        ========================
        parameters: dict
            LightGBM parameters.
        rounds: int
            Number of boosting rounds.
        """
        self.parameters = parameters if parameters else {}
        self.rounds = rounds or 10
        self.booster = None


    def train(self, train_dataset, val_dataset):
        print("lgb parameters:")
        print(self.parameters)
        self.booster = lgb.train(
            params=self.parameters,
            train_set=train_dataset,
            valid_sets=val_dataset,
            num_boost_round=self.rounds,
        )

    def predict(self, X, **kwargs):
        """Predict using the booster on transformed features."""
        return self.booster.predict(X, **kwargs)
