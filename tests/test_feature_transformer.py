import pandas as pd
from gbt import FeatureTransformer


def test_convert_categorical_features():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
    transformer = FeatureTransformer(categorical_features=["b"], target="a")
    transformer.fit(df)
    converted = transformer.convert_categorical_features(df)
    print(converted)


def test_feature_transformer_can_save_and_load(tmpdir):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
    transformer = FeatureTransformer(
        categorical_features=["b"], target="a", output_dir=str(tmpdir)
    )
    transformer.fit(df)
    transformer.save()
    assert (tmpdir / "feature_transformer.json").exists()
    loaded = FeatureTransformer.from_json(str(tmpdir))
    assert transformer.categorical_features == loaded.categorical_features
    assert transformer.target == loaded.target
    assert transformer.output_dir == loaded.output_dir
    assert transformer.numerical_features == loaded.numerical_features
