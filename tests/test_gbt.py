from os import path

import pandas as pd
from gbt import train, load, TrainingPipeline


csd = path.dirname(path.realpath(__file__))


def test_model_can_train():
    train(
        path.join(csd, "test_data", "1.csv"),
        categorical_feature_columns=["col2"],
        numerical_feature_columns=["col1"],
        sort_by_columns=None,
        label_column="label",
    )


def test_model_can_train_with_pipeline():
    class DatasetBuilder:
        def training_dataset(self):
            return path.join(csd, "test_data", "1.csv")

    TrainingPipeline(
        categorical_feature_columns=["col2"],
        numerical_feature_columns=["col1"],
        sort_by_columns=None,
        label_column="label",
    ).fit(DatasetBuilder())


def test_the_readme_example():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": ["a", "b", "c", None, "e", "f", "g"],
            "c": [1, 0, 1, 1, 0, 0, 1],
            "some_other_column": [0, 0, None, None, None, 3, 3],
        }
    )
    train(
        df,
        model_lib="binary",
        label_column="c",
        val_size=0.2,  # fraction of the validation split
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
    )


def test_predict_after_training():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": [1, 0, 1],
        }
    )
    pipeline = train(
        df,
        model_lib="binary",
        label_column="c",
        val_size=0.2,
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
    )
    preds = pipeline.predict(df)
    assert len(preds) == len(df)


def test_pipeline_can_save_and_load(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": [1, 0, 1],
        }
    )
    train(
        df,
        model_lib="binary",
        label_column="c",
        val_size=0.2,
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
        log_dir=str(tmp_path),
    )
    loaded = load(str(tmp_path))
    new_df = df.drop(columns=["c"])
    preds = loaded.predict(new_df)
    assert len(preds) == len(new_df)


def test_the_readme_example_with_pipeline():
    class DatasetBuilder:
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
        params_preset="binary",
        params_override={"num_leaves": 10},
        label_column="c",
        val_size=0.2,  # fraction of the validation split
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
    ).fit(DatasetBuilder())
