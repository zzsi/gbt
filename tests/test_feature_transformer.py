import pandas as pd
from gbt.feature_transformer import FeatureTransformer


def test_convert_categorical_features():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
    transformer = FeatureTransformer(categorical_features=["b"], target="a")
    transformer.fit(df)
    converted = transformer.convert_categorical_features(df)
    print(converted)