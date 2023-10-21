`gbt` is a library for gradient boosted trees with minimal coding required. It is a thin wrapper around [`lightgbm`](https://lightgbm.readthedocs.io/). Give it a `pandas.Dataframe`, `gbt.train()` takes care of feature transforms (e.g. scaling for numerical features, label encoding for categorical features) and metrics print outs.

## Install

```
pip install gbt
```


## Quickstart

```python
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
    params_preset="binary",  # one of mape, l2, binary, multiclass
    params_override={"num_leaves": 10},
    label_column="c",
    val_size=0.2,  # fraction of the validation split
    categorical_feature_columns=["b"],
    numerical_feature_columns=["a"],
).fit(DatasetBuilder())
```

