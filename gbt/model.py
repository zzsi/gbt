from os import path, makedirs
from typing import Union
import pandas as pd
import lightgbm as lgb
from .feature_transformer import FeatureTransformer


class GBTModel:
    """Minimal inference-only model containing booster + feature transformer."""
    
    def __init__(self, booster, feature_transformer):
        """
        Initialize GBTModel with trained components.
        
        Args:
            booster: Trained LightGBM booster
            feature_transformer: Fitted FeatureTransformer
        """
        self.booster = booster
        self.feature_transformer = feature_transformer
    
    def predict(self, df: Union[str, pd.DataFrame], **kwargs):
        """
        Predict on new data.
        
        Args:
            df: Input dataframe or path to CSV file
            **kwargs: Additional arguments passed to booster.predict()
            
        Returns:
            Predictions array
        """
        # Handle file path input
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame or str, got {type(df)}")
            
        # Transform features and predict
        features = self.feature_transformer.transform(df)
        return self.booster.predict(features, **kwargs)
    
    def save(self, model_dir: str):
        """
        Save model artifacts for inference.
        
        Args:
            model_dir: Directory to save artifacts
        """
        makedirs(model_dir, exist_ok=True)
        
        # Save booster
        model_path = path.join(model_dir, "lgb_classifier.txt")
        self.booster.save_model(model_path)
        
        # Save feature transformer
        transformer_path = path.join(model_dir, "feature_transformer.json")
        self.feature_transformer.output_dir = model_dir
        self.feature_transformer.save()
    
    @classmethod
    def load(cls, model_dir: str) -> "GBTModel":
        """
        Load model for inference.
        
        Args:
            model_dir: Directory containing saved artifacts
            
        Returns:
            GBTModel instance ready for inference
        """
        model_path = path.join(model_dir, "lgb_classifier.txt")
        transformer_path = path.join(model_dir, "feature_transformer.json")
        
        # Check if files exist
        if not path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not path.exists(transformer_path):
            raise FileNotFoundError(f"Feature transformer not found: {transformer_path}")
        
        # Load booster
        booster = lgb.Booster(model_file=model_path)
        
        # Load feature transformer
        feature_transformer = FeatureTransformer.from_json(model_dir)
        
        return cls(booster, feature_transformer)