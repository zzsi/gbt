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
from gbt import train

# Your data
df = pd.DataFrame({
    "feature_1": [1, 2, 3, 4, 5],
    "category": ["A", "B", "A", "C", "B"], 
    "target": [0, 1, 0, 1, 1]
})

# Train model
model = train(
    df,
    model_lib="binary",  # binary, multiclass, mape, or l2
    label_column="target",
    categorical_feature_columns=["category"],
    numerical_feature_columns=["feature_1"]
)

# Make predictions
new_data = pd.DataFrame({
    "feature_1": [6, 7],
    "category": ["A", "B"]
})
predictions = model.predict(new_data)
print(predictions)  # [0.23, 0.78]
```

## Save and Load Models

```python
# Save model
model.save("my_model")

# Load model later
from gbt import load
loaded_model = load("my_model")
predictions = loaded_model.predict(new_data)
```

## Advanced Usage

For more control over training:

```python
from gbt import TrainingPipeline

# Custom training configuration  
pipeline = TrainingPipeline(
    categorical_feature_columns=["category"],
    numerical_feature_columns=["feature_1"],
    params_preset="binary",
    params_override={"num_leaves": 50},  # Custom hyperparameters
    val_size=0.3,  # 30% validation split
    verbose=False   # Quiet training
)

# Train with custom data loader
class DatasetBuilder:
    def training_dataset(self):
        return pd.read_csv("train.csv")
    
    def testing_dataset(self):
        return pd.read_csv("test.csv")

pipeline.fit(DatasetBuilder())

# Get model for deployment
model = pipeline.create_model()
model.save("production_model")
```

## API Reference

### Main Functions

#### `train(df, model_lib="l2", label_column, categorical_feature_columns, numerical_feature_columns, **kwargs)`

Train a gradient boosting model.

**Parameters:**
- `df`: Training dataframe
- `model_lib`: Model type - `"binary"`, `"multiclass"`, `"mape"`, or `"l2"`
- `label_column`: Target column name  
- `categorical_feature_columns`: List of categorical feature names
- `numerical_feature_columns`: List of numerical feature names
- `val_size`: Validation split fraction (default 0.2)
- `log_dir`: Directory to save artifacts (optional)

**Returns:** Trained model ready for prediction

#### `load(path)`

Load a saved model.

**Returns:** Model ready for inference

### Model Types

| `model_lib` | Use Case | Loss Function |
|-------------|----------|---------------|
| `"binary"` | Binary classification | Log loss |
| `"multiclass"` | Multi-class classification | Multi-class log loss |
| `"l2"` | Regression | Mean squared error |
| `"mape"` | Regression | Mean absolute percentage error |

### Model Methods

- `.predict(df)`: Make predictions
- `.save(path)`: Save model to disk

For advanced usage, see `TrainingPipeline` class documentation.
