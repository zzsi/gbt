`gbt` is a library for gradient boosted trees with minimal coding required. It is a thin wrapper around [`lightgbm`](https://lightgbm.readthedocs.io/).

What you need:
- a pandas dataframe,
- the target column to predict on,
- categorical feature columns (can be empty),
- numerical feature columns (can be empty, but you should have at least one categorical or numerical feature),
- params_preset: "binary", "multiclass", "mape", "l2" to specify what type of prediction objective and default hyperparameters to use.

You don't need to (though you are welcome to):
- normalize the numerical feature values
- construct the encoder to one-hot encode categorical features
- manage saving of artifacts for above feature transformation
- implement evaluation metrics


## Install

```
pip install gbt
```


## Quickstart

```python

train_df = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6, 7],
        "b": ["a", "b", "c", None, "e", "f", "g"],
        "c": [1, 0, 1, 1, 0, 0, 1],
        "some_other_column": [0, 0, None, None, None, 3, 3],
    }
)

class DatasetBuilder:
    def training_dataset(self):
        return train_df
    
    def testing_dataset(self):
        return train_df  # replace with a real test dataset in practice

TrainingPipeline(
    params_preset="binary",  # one of mape, l2, binary, multiclass
    params_override={"num_leaves": 10},
    label_column="c",
    val_size=0.2,  # fraction of the validation split
    categorical_feature_columns=["b"],
    numerical_feature_columns=["a"],
).fit(DatasetBuilder())
```

## Output of training

The output includes:

- the model file (decision trees and boosting coefficients),
- a feature transformer state file (in JSON)

## Using the model to predict on new data
Once a model is trained, the returned pipeline can be used for
prediction on any dataframe with the same schema.

```python
# Train and save artifacts to "my_model" directory
from gbt import train, load

pipeline = train(
    df,
    model_lib="binary",
    label_column="c",
    categorical_feature_columns=["b"],
    numerical_feature_columns=["a"],
    val_size=0.2,
    log_dir="my_model",
)

new_df = pd.DataFrame({"a": [8, 9], "b": ["h", "i"]})
predictions = pipeline.predict(new_df)

# Later on, reload the pipeline for inference
loaded_pipeline = load("my_model")
predictions = loaded_pipeline.predict(new_df)
```
