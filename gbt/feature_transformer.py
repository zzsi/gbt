from typing import List, Optional

import json
from os import path
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class FeatureTransformer:
    """A trainable transformer which preprocess numerical
    and categorical features.

    This class is used both in training and testing. The trainable
    parameters (normalization and scaling parameters, categorical
    vocabularies) are saved during training and loaded during prediction.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        target: Optional[str] = None,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        add_categorical_stats=False,
        order_categoricals=False,
        drop_categoricals=False,
        preprocess_fn=None,
        postprocess_fn=None,
        count_vectorize=False,
        apply_vectorize=None,
    ):
        """
        Params
        =========
        add_categorical_stats: bool, default False
            If True, add derived features each categorical variable,
            computed as mean and count (and other stats) of the target value per level.
            The stats could be computed in a windowing fashion.
        """
        # TODO: rename target to label_column
        self.target = target
        assert self.target is not None, "No target column. What should I predict?"
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.order_categoricals = order_categoricals
        if len(self.categorical_features) == 0 and len(self.numerical_features) == 0:
            raise ValueError(
                "No categorical features, no numerical features. What am I supposed to do?"
            )
        self.cats = {}
        self.aggs = {}
        self.output_dir = output_dir
        if self.output_path and path.exists(self.output_path):
            self.load()
        self.add_categorical_stats = add_categorical_stats
        self.drop_categoricals = drop_categoricals
        # TODO: not being able save preprocess_fn, postprocess_fn can be problem
        #   if these functions change frequently.
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.count_vectorize = count_vectorize
        self.apply_vectorize = apply_vectorize
        self.features = []

    @classmethod
    def from_json(cls, json_path: str) -> "FeatureTransformer":
        """
        Load a FeatureTransformer from a json file.

        Args:
            json_path (str): Path to the json file or the directory that has a feature_transformer.json.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"File or directory {json_path} does not exist.")
        if json_path.is_dir():
            json_path = json_path / "feature_transformer.json"
        with open(str(json_path)) as f:
            obj = json.load(f)
            parent_dir = json_path.parent
            feature_transfomer = cls(
                output_dir=str(parent_dir),
                target=obj["target"],
                categorical_features=obj["categorical_features"],
                numerical_features=obj["numerical_features"],
                add_categorical_stats=obj["add_categorical_stats"],
                order_categoricals=obj.get("order_categoricals", False),
                drop_categoricals=obj.get("drop_categoricals", False),
                # TODO: not being able save preprocess_fn, postprocess_fn can be problem
            )
            feature_transfomer.cats_encoder = {}
            for feature_name, vocab in feature_transfomer.cats.items():
                feature_transfomer.cats_encoder[feature_name] = {
                    value: i for i, value in enumerate(vocab)
                }
            print(f"Feature transformer loaded from {feature_transfomer.output_path}")
            return feature_transfomer

    def fit(self, df):
        if self.preprocess_fn is not None and callable(self.preprocess_fn):
            df = self.preprocess_fn(df)
        if self.add_categorical_stats:
            self.aggs = {}
            for f in self.categorical_features:
                agg = df.groupby(f)[self.target].agg(["mean", "count"]).do_ingest_job()
                agg = agg.rename(
                    columns={
                        "mean": self.derived_mean_feature_name(f),
                        "count": self.derived_count_feature_name(f),
                    },
                    # inplace=True,
                )
                self.aggs[f] = agg
        for f in self.categorical_features:
            if self.order_categoricals:
                series = pd.Series(
                    pd.Categorical(df[f], ordered=self.order_categoricals)
                )
            else:
                series = df[f].astype("category")
            self.cats[f] = series.cat.categories

    def fit_nlp_features(self, df):
        pass

    def derived_mean_feature_name(self, feature_name):
        return f"avg_{self.target}_for_same_{feature_name}"

    def derived_count_feature_name(self, feature_name):
        return f"count_for_same_{feature_name}"

    @property
    def feature_names(self):
        if self.drop_categoricals:
            return self.numerical_features + self.derived_features
        else:
            return (
                self.categorical_features
                + self.numerical_features
                + self.derived_features
            )

    @property
    def derived_features(self):
        if self.add_categorical_stats:
            return [
                self.derived_mean_feature_name(f) for f in self.categorical_features
            ] + [self.derived_count_feature_name(f) for f in self.categorical_features]
        else:
            return []

    def transform(
        self, df, include_target_column=False, include_original_columns=False
    ):
        # Please add additional feature normalization and preprocessing here.

        # Make sure categoricals are pd.category and numericals are np.float.
        if self.preprocess_fn is not None and callable(self.preprocess_fn):
            df = self.preprocess_fn(df)

        if self.add_categorical_stats:
            for f, agg in self.aggs.items():
                df = df.merge(agg, on=f, how="left", suffixes=(None, "_y"))

        self.convert_categorical_features(df)
        self.convert_numerical_features(df)
        self.transform_nlp_features(df)

        features = self.feature_names
        if self.count_vectorize:  # check if its training and calculate cv object
            df, cv = self.count_vectorize_feature(df)
            self.apply_vectorize = cv
            features.remove("Variety")  # removing variety
            for a in list(cv.vocabulary_):
                features.append(a)  # adding the 200 features back to features
        if self.apply_vectorize and not self.count_vectorize:  # check for validation
            df = self.apply_count_vectorize_to_val(self.apply_vectorize, df)
            features.remove("Variety")
            for a in list(self.apply_vectorize.vocabulary_):
                features.append(a)
            self.features = features  # initializing so it could be used in testing

        if self.postprocess_fn is not None and callable(self.postprocess_fn):
            df = self.postprocess_fn(df)

        if include_target_column and self.target in df.columns:
            self.convert_labels(df)
            features.append(self.target)
        if include_original_columns:
            return df
        else:
            return df[features]

    def transform_nlp_features(self, df):
        pass

    def count_vectorize_feature(self, df):
        """
        Initialize count vactorize and concatenate cv dataframe with initialized features and our orginal dataframe
        """
        cv = CountVectorizer(
            max_features=200,
            stop_words=["all", "in", "the", "is", "and", "a", "an"],
        )
        df = self.convert_categorical_to_str(df)
        count_vector = cv.fit_transform(df["Variety"])
        df_train_merged = self.count_vectorize_merge(count_vector, df, cv)
        return df_train_merged, cv

    def apply_count_vectorize_to_val(self, cv, df):
        """
        apply the cv from the training and concatenate the initial df with our orginal df
        """
        df = self.convert_categorical_to_str(df)
        count_vector_val = cv.transform(df["Variety"])
        df_val_merged = self.count_vectorize_merge(count_vector_val, df, cv)
        return df_val_merged

    def count_vectorize_merge(self, count_vector, df, cv):
        """
        concatenating and dropping variety since we are adding cv features extracted from variety
        """
        data_frame = pd.DataFrame(count_vector.toarray(), columns=list(cv.vocabulary_))
        df = df.drop(["Variety"], axis=1)
        df = df.do_ingest_job()
        df_merged = pd.concat([df, data_frame], axis=1)
        df_merged = df_merged.drop(["index"], axis=1)
        return df_merged

    def convert_categorical_to_str(self, df):
        if df["Variety"].dtype == "category":
            df["Variety"] = df["Variety"].astype(str)
            return df
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    @property
    def output_path(self):
        if self.output_dir is None:
            return None
        return path.join(self.output_dir, "feature_transformer.json")

    def save(self):
        aggs = {}
        for field, stats_df in self.aggs.items():
            aggs[field] = stats_df.to_dict()
        with open(self.output_path, "w") as f:
            json.dump(
                dict(
                    target=self.target,
                    categorical_features=self.categorical_features,
                    numerical_features=self.numerical_features,
                    cats=self.cats,
                    aggs=aggs,
                    add_categorical_stats=self.add_categorical_stats,
                    drop_categoricals=self.drop_categoricals,
                ),
                f,
                indent=2,
                cls=NumpyEncoder,
            )
            print(f"Feature transformer saved to {self.output_path}")

    def load(self):
        """
        Obsolete. Use from_json instead.
        """
        with open(self.output_path) as f:
            obj = json.load(f)
            for f in [
                "target",
                "categorical_features",
                "numerical_features",
                "cats",
            ]:
                setattr(self, f, obj[f])
            self.cats_encoder = {}
            for feature_name, vocab in self.cats.items():
                self.cats_encoder[feature_name] = {
                    value: i for i, value in enumerate(vocab)
                }
            print(f"Feature transformer loaded from {self.output_path}")

    def convert_numerical_features(self, df):
        for f in self.numerical_features:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    def convert_categorical_features(self, df):
        for f in self.categorical_features:
            if self.order_categoricals:
                df[f] = df[f].astype("category")
                df[f] = df[f].cat.set_categories(self.cats[f])  # , inplace=True)
                df[f] = df[f].astype(float)
            else:
                df[f] = df[f].astype("category")
                df[f] = df[f].cat.set_categories(self.cats[f])  # , inplace=True)

    def convert_labels(self, df):
        df[self.target] = df[self.target].astype(float)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, pd.Index) or o.__class__.__name__ == "Index":
            return o.tolist()
        return json.JSONEncoder.default(self, o)
