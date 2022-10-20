from os import path

import pandas as pd
from gbt.api import train


csd = path.dirname(path.realpath(__file__))


def test_model_can_train():
    train(
        path.join(csd, "test_data", "1.csv"),
        categorical_feature_columns=["col2"],
        numerical_feature_columns=["col1"],
        sort_by_columns=None,
        label_column="label",
    )


def test_the_readme_example():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7],
        "b": ["a", "b", "c", None, "e", "f", "g"],
        "c": [1, 0, 1, 1, 0, 0, 1],
        "some_other_column": [0, 0, None, None, None, 3, 3]
    })
    train(
        df,
        recipe="binary",
        label_column="c",
        val_size=0.2,  # fraction of the validation split
        categorical_feature_columns=["b"],
        numerical_feature_columns=["a"],
    )