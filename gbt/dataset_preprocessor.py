from glob import glob
from os import makedirs, path
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import Dataset


class DataPreprocessor:
    def __init__(
        self,
        local_dir_or_file="data/csv",
        df=None,
        log_dir=None,
        nrows=None,
        feature_transformer=None,
        sort_by_columns=None,
        label_column="label",
    ):
        """
        DataPreprocessor is a class to preprocess the data for training.
        It handles the following tasks:
        - read the data from local file or directory.
        - sort the data by columns.
        - split the data into train/val.
        - feature engineering.
        - save the feature transformer.


        Args:
            local_dir_or_file: str, local dir or file path.
            df: pd.DataFrame, if not None, will ignore local_dir_or_file.
            log_dir: str, path to the directory to save the feature transformer.
            nrows: int, number of rows to read from the csv file.
            feature_transformer: FeatureTransformer, if None, will create a default one.
            sort_by_columns: list of str, columns to sort by.
            label_column: str, name of the label column.

        Returns:
            DataPreprocessor



        """
        self.log_dir = log_dir
        if log_dir is not None:
            makedirs(log_dir, exist_ok=True)
        self.nrows = nrows
        self.feature_transformer = feature_transformer
        self.label_column = label_column
        self.sort_by_columns = sort_by_columns
        if local_dir_or_file is not None and Path(local_dir_or_file).is_file():
            self.df = self.read_csv(local_dir_or_file)
        elif local_dir_or_file is not None and path.isdir(local_dir_or_file):
            self.local_dir = local_dir_or_file
            self.dfs = {}
            glob_pattern = path.join(self.local_dir, "*.csv*")
            print(f"Looking for: {glob_pattern}")
            for fn in glob(glob_pattern):
                key = path.basename(fn).rstrip(".csv.gz").rstrip(".csv")
                self.dfs[key] = self.read_csv(fn)
            self.df = pd.concat(list(self.dfs.values()))
        elif local_dir_or_file is None:
            self.df = df
        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()

    def read_csv(self, path):
        return pd.read_csv(
            path,
            dtype=str,
            nrows=self.nrows,
        )

    def preprocess(self):
        # TODO: sort_by_columns rename to sortby_columns. Allow desc, asc.
        if self.sort_by_columns is not None and len(self.sort_by_columns) > 0:
            self.df.sort_values(by=self.sort_by_columns, inplace=True)

        self.features = self.df
        self.labels = self.df[self.label_column].astype(float)

    def split(
        self,
        val_size=0.2,
        pretrain_size=0.5,
        shuffle=True,
        random_state=0,
        lgb_data=True,
        pd_data=False,
    ):
        """Random split the fraction of val_size as validation set.

        TODO: implement more splitters.

        pretrain_size: among training data, how much to allocate for pre-train.
        """
        if not shuffle:
            print("Not doing random shuffling in splitting..")

        (train_x, val_x, train_y, val_y) = train_test_split(
            self.features,
            self.labels,
            test_size=val_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        if (pretrain_size or 0) > 0:
            pretrain_x, train_x, _, train_y = train_test_split(
                train_x,
                train_y,
                test_size=1 - pretrain_size,
                random_state=random_state,
                shuffle=shuffle,
            )
            self.pretrain_features = pretrain_x
            self.pretrain_y = pretrain_x[self.label_column]
        else:
            self.pretrain_features = train_x
            self.pretrain_y = train_y

        self.train_features = train_x
        self.train_labels = train_y
        self.val_features = val_x
        self.val_labels = val_y

        if self.log_dir and path.exists(
            path.join(self.log_dir, "feature_transformer.json")
        ):
            self.feature_transformer.load()
        else:
            # TODO: if feature_transformer is None, create a dummy one.
            if hasattr(self, "pretrain_features"):
                self.feature_transformer.fit(self.pretrain_features)
                if self.log_dir is not None:
                    self.feature_transformer.save()
        self.feature_transformer_is_fitted = True
        self.original_df = self.df
        self.features = self.feature_transformer.transform(
            self.df.copy(), include_target_column=True
        )
        self.labels = self.features[self.label_column]
        self.features = self.features.drop(columns=[self.label_column])
        self.feature_transformer.count_vectorize = False  # setting to true for training
        self.train_features = self.feature_transformer.transform(
            self.train_features, include_target_column=True
        )
        self.train_labels = self.train_features[self.label_column]
        self.train_features = self.train_features.drop(columns=[self.label_column])
        # setting false indicating we finished training
        self.feature_transformer.count_vectorize = False
        self.val_features = self.feature_transformer.transform(
            self.val_features,
            include_target_column=True,
        )
        self.val_labels = self.val_features[self.label_column]
        self.val_features = self.val_features.drop(columns=[self.label_column])

        if lgb_data:
            train_dataset = lgb.Dataset(
                data=self.train_features[self.feature_transformer.feature_names],
                label=self.train_labels,
                free_raw_data=False,
            )
            val_dataset = lgb.Dataset(
                data=self.val_features[self.feature_transformer.feature_names],
                label=self.val_labels,
                reference=train_dataset,
                free_raw_data=False,
            )
        elif pd_data:
            train_dataset = Dataset(
                features=self.train_features, target=self.train_labels
            )
            val_dataset = Dataset(features=self.val_features, target=self.val_labels)
        else:
            raise ValueError("Neither lgb_data or pd_data is True?")

        return train_dataset, val_dataset

    def get_embedding_data(self, columns):
        return self.df[columns]

    def lgb_dataset(self):
        return lgb.Dataset(self.features, label=self.labels)
