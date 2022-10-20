`gbt` is a library for gradient boosted trees with minimal coding required. It is a thin wrapper around [`lightgbm`](https://lightgbm.readthedocs.io/en/v3.3.2/). Give it a `pandas.Dataframe`, `gbt.train()` takes care of feature transforms (e.g. scaling for numerical features, label encoding for categorical features) and metrics print outs.

Example usage:

```
import pandas as pd
import gbt

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
```

Supported "recipes": mape, l2, l2_rf, binary, multiclass.
