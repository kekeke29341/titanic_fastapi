import os
import joblib
from typing import Dict, Any
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModel:
    """
    Service for making predictions using the LightGBM model.
    """
    
    def __init__(self):
        """Initialize the ML model service."""
        self.model = None
        self.preprocessor = None
        self._model_loaded = False
        self.model_dir = os.environ.get(
            "MODEL_DIR", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml/model_artifacts")
        )
        
    def load_model(self):
        """Load the model and preprocessor from disk."""
        try:
            logger.info("Loading model and preprocessor...")
            self.model = joblib.load(os.path.join(self.model_dir, "model.pkl"))
            self.preprocessor = joblib.load(os.path.join(self.model_dir, "preprocessor.pkl"))
            self._model_loaded = True
            logger.info("Model and preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the loaded model.
        
        Args:
            features: Dictionary of passenger features
            
        Returns:
            Dictionary with prediction result and probability
        """
        if not self._model_loaded:
            logger.info("Model not loaded yet, loading now...")
            self.load_model()
        
        try:
            # Preprocess the features
            X = self._preprocess_features(features)
            
            # Make prediction
            prob = self.model.predict_proba(X)[0, 1]  # Probability of class 1 (survived)
            prediction = 1 if prob >= 0.5 else 0
            
            return {
                "prediction": prediction,
                "probability": float(prob)
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the input features using the saved preprocessor.
        
        Args:
            features: Dictionary of passenger features
            
        Returns:
            Preprocessed features as numpy array
        """
        # Convert to DataFrame-like structure for preprocessing
        X = {k: [v] for k, v in features.items()}
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform(X)
        
        return X_processed
