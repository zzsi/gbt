`gbt` is a library for gradient boosted trees with minimal coding required.

Example usage:

```
import pandas as pd
import gbt

df = pd.read_csv("my_data.csv")
gbt.train(
    df,
    target_column=...,
    val_size=0.2,  # fraction of the validation split
    categorical_feature_columns=[],
    numerical_feature_columns=[],
)
```
