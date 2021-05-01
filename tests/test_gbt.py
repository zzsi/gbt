from os import path

from boosted_tree_kungfu.api import train

csd = path.dirname(path.realpath(__file__))


def test_model_can_train():
    train(
        path.join(csd, "test_data", "1.csv"),
        categorical_feature_columns=["col2"],
        numerical_feature_columns=["col1"],
        sort_by_columns=None,
        label_column="label",
    )
