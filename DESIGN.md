# GBT Architecture Design

## Overview

This document outlines the architectural design principles and decisions for the `gbt` library, particularly focusing on the separation of training and inference concerns.

## Design Principles

### 1. Separation of Concerns
**Training** and **inference** have fundamentally different requirements:

| Training | Inference |
|----------|-----------|
| Stateful process | Stateless function |
| Complex configuration | Minimal configuration |
| Needs metrics, logging, callbacks | Needs speed and simplicity |
| Modifies state iteratively | Pure transformation |
| Development/experimentation focus | Production/deployment focus |

### 2. Learning from Other Frameworks

#### PyTorch Lightning Pattern
```python
# Model defines computation
class Model(pl.LightningModule):
    def training_step(self, batch): ...

# Trainer orchestrates training
trainer = Trainer()
trainer.fit(model)  # Modifies model in-place
```
**Key insight**: Trainer doesn't return a model; it modifies the model in-place.

#### HuggingFace Pattern
```python
# Training with Trainer
trainer = Trainer(model=model, args=args)
trainer.train()  # Returns metrics, not model

# Inference with Model
model = AutoModel.from_pretrained("path")
outputs = model(inputs)
```
**Key insight**: Clear separation between training orchestration and model usage.

#### sklearn Pattern
```python
pipeline = Pipeline([('scaler', StandardScaler()), ('model', SVC())])
pipeline.fit(X, y)  # Modifies in-place
predictions = pipeline.predict(X)
```
**Key insight**: Pipeline owns the entire transformation + model state.

### 3. State Management Philosophy

Training state is complex and includes:
- Model parameters (needed for inference)
- Optimizer state (training only)
- Learning rate schedules (training only)
- Metrics history (training only)
- Best iteration tracking (training only)
- Data statistics (training only)

Inference only needs:
- Model parameters
- Preprocessing configuration

## Architecture

### GBTModel
```python
class GBTModel:
    """Minimal inference-only class"""
    def __init__(self, booster, feature_transformer):
        self.booster = booster  # LightGBM booster
        self.feature_transformer = feature_transformer
    
    def predict(self, df):
        features = self.feature_transformer.transform(df)
        return self.booster.predict(features)
    
    @classmethod
    def load(cls, path):
        # Load only inference artifacts
        pass
    
    def save(self, path):
        # Save only inference artifacts
        pass
```

### TrainingPipeline
```python
class TrainingPipeline:
    """Training orchestrator"""
    def fit(self, dataset_builder):
        # Training logic
        # Updates self.model and self.feature_transformer
        pass
    
    def create_model(self) -> GBTModel:
        """Export minimal inference object"""
        return GBTModel(self.model.booster, self.feature_transformer)
    
    def save_checkpoint(self, path):
        """Save full training state for resuming"""
        # Include optimizer state, metrics, config, etc.
        pass
```

### LightGBMModel
```python
class LightGBMModel:
    """Pure LightGBM booster wrapper"""
    def __init__(self, parameters, rounds):
        self.booster = None
        # No feature_transformer!
    
    def train(self, train_ds, val_ds):
        self.booster = lgb.train(...)
    
    def predict(self, X):
        return self.booster.predict(X)
```

## Usage Patterns

### Training Phase
```python
# Configure training
pipeline = TrainingPipeline(
    categorical_features=["cat1"],
    numerical_features=["num1"],
    params_preset="binary",
    val_size=0.2,
    early_stopping_rounds=10
)

# Train
pipeline.fit(data)

# Export for deployment
model = pipeline.create_model()
model.save("model/")
```

### Inference Phase
```python
# Load minimal model
model = GBTModel.load("model/")

# Predict
predictions = model.predict(new_data)
```

### Backward Compatibility
```python
# Old API still works
pipeline = train(df, ...)  # Returns TrainingPipeline
predictions = pipeline.predict(df)  # Works, but internally uses GBTModel

# New API for deployment
model = pipeline.create_model()  # Get clean inference object
```

## Implementation Considerations

### Migration Strategy
1. **Phase 1**: Add GBTModel without breaking changes
2. **Phase 2**: Deprecate TrainingPipeline.predict() in favor of GBTModel
3. **Phase 3**: Update load() to return GBTModel by default

### Backward Compatibility
- Keep existing API working
- Add deprecation warnings gradually
- Provide clear migration examples

### File Structure
```
gbt/
  model.py           # New: GBTModel for inference
  training_pipeline.py  # Refactored: Training only
  lightgbm_model.py    # Simplified: Pure booster wrapper
  api.py              # Updated: Public API
```

## Benefits of This Design

1. **Clear Responsibilities**: Each class has a single, clear purpose
2. **Deployment Ready**: GBTModel is minimal and production-friendly
3. **Extensible**: Easy to add training features without affecting inference
4. **Testable**: Can test inference without training dependencies
5. **MLOps Friendly**: Clean artifacts for model versioning and deployment

## Future Extensions

This design enables:
- Integration with experiment tracking (W&B, MLflow)
- Model versioning and registry
- A/B testing different models
- Distributed training without affecting inference API
- Custom callbacks and plugins for training
- Model serving optimizations

## Decision Record

**Decision**: Use `GBTModel` as the inference class name
- **Rationale**: Simple, clear, follows industry conventions
- **Alternatives considered**: GBTInferenceModel (too verbose), GBTInferencePipeline (confusing with TrainingPipeline), GBTPredictor (less common)

**Decision**: TrainingPipeline.fit() doesn't return a model
- **Rationale**: Follows modern ML framework patterns, training modifies state in-place
- **Alternatives considered**: Returning a new model (wasteful, not how training works)

**Decision**: Keep preprocessing in the inference model
- **Rationale**: Preprocessing is part of the model's computation graph for deployment
- **Alternatives considered**: Separate preprocessor (more complex deployment)