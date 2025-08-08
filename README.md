`gbt` is a library for gradient boosted trees with minimal coding required. It is a thin wrapper around [`lightgbm`](https://lightgbm.readthedocs.io/).

## Features

- Zero feature engineering needed - automatic encoding of categorical features
- Built-in train/validation splitting
- Automatic artifact saving and loading
- Pre-configured model presets for common tasks (binary, multiclass, regression)

What you need:
- a pandas dataframe,
- the target column to predict on,
- categorical feature columns (can be empty),
- numerical feature columns (can be empty, but you should have at least one categorical or numerical feature),
- model_lib: "binary", "multiclass", "mape", "l2" to specify what type of prediction objective and default hyperparameters to use.

You don't need to (though you are welcome to):
- normalize the numerical feature values
- construct the encoder to one-hot encode categorical features
- manage saving of artifacts for above feature transformation
- implement evaluation metrics


## Prerequisites

- Python 3.7+
- pandas, numpy, scikit-learn, lightgbm

## Install

```
pip install gbt
```


## Quickstart

```python
import pandas as pd
from gbt import TrainingPipeline

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

### Simple API (Recommended)

```python
# Train and save artifacts
from gbt import train, load

pipeline = train(
    df,
    model_lib="binary",  # one of binary, multiclass, mape, l2
    label_column="c",
    categorical_feature_columns=["b"],
    numerical_feature_columns=["a"],
    val_size=0.2,
    log_dir="my_model",
)

# Option 1: Use pipeline directly (backward compatible)
predictions = pipeline.predict(new_df)

# Option 2: Get clean model for deployment (recommended)
model = pipeline.create_model()
predictions = model.predict(new_df)
model.save("my_model")

# Later: Load model for inference
from gbt import GBTModel
model = GBTModel.load("my_model")  # or use load("my_model")
predictions = model.predict(new_df)
```

### Advanced Usage

```python
# For more control during training
from gbt import TrainingPipeline

pipeline = TrainingPipeline(
    categorical_feature_columns=["b"],
    numerical_feature_columns=["a"],
    params_preset="binary",
    val_size=0.2,
    verbose=False  # Suppress training output
)
pipeline.fit(dataset_builder)

# Export clean model for production
model = pipeline.create_model()
model.save("production_model")
```

## API Reference

### Training

#### `train(df, model_lib="l2", label_column, categorical_feature_columns, numerical_feature_columns, val_size=0.2, log_dir=None)`

Train a model with simplified API.

- `df`: Training dataframe
- `model_lib`: One of "binary", "multiclass", "mape", "l2" 
- `label_column`: Target column name
- `categorical_feature_columns`: List of categorical feature names
- `numerical_feature_columns`: List of numerical feature names
- `val_size`: Validation split fraction (default 0.2)
- `log_dir`: Directory to save model artifacts
- Returns: `TrainingPipeline` instance

#### `TrainingPipeline`

- `.fit(dataset_builder)`: Train the model
- `.create_model()`: Export `GBTModel` for inference
- `.predict(df)`: Predict (backward compatible)

### Inference

#### `GBTModel`

Minimal model for production inference.

- `.predict(df)`: Make predictions on new data
- `.save(path)`: Save model artifacts
- `.load(path)`: Load model from artifacts (class method)

#### `load(log_dir)`

Load a saved model for inference. Returns `GBTModel` instance.

### Model Presets

- `"binary"`: Binary classification with log loss
- `"multiclass"`: Multi-class classification
- `"mape"`: Regression with MAPE objective
- `"l2"`: Regression with L2/MSE objective (default)
