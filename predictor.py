"""
Fraud Predictor - Loads model and makes predictions
"""
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import time


class FraudPredictor:
    def __init__(self, model_path, scaler_path, features_path, imputer_path, config):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, "rb") as f:
            self.feature_names = pickle.load(f)
        
        with open(imputer_path, "rb") as f:
            self.imputer = pickle.load(f)
        
        # Get original feature names from imputer (before feature selection)
        self.original_features = self.imputer.feature_names_in_
        
        self.fraud_threshold = config.get("FRAUD_THRESHOLD", 0.5)
        self.risk_low = config.get("RISK_LOW", 0.30)
        self.risk_medium = config.get("RISK_MEDIUM", 0.60)
        self.risk_high = config.get("RISK_HIGH", 0.85)
    
    def predict(self, transaction_data):
        """Make prediction on a single transaction."""
        start_time = time.time()
        
        # Step 1: Create DataFrame with ORIGINAL features (before feature selection)
        original_data = {feat: 0.0 for feat in self.original_features}
        
        # Update with provided values
        for key, value in transaction_data.items():
            if key in original_data:
                original_data[key] = value
        
        # Create DataFrame with original features in correct order
        df_original = pd.DataFrame([original_data], columns=self.original_features)
        
        # Step 2: Impute missing values
        df_imputed = self.imputer.transform(df_original)
        df_imputed = pd.DataFrame(df_imputed, columns=self.original_features)
        
        # Step 3: Scale features
        df_scaled = self.scaler.transform(df_imputed)
        df_scaled = pd.DataFrame(df_scaled, columns=self.original_features)
        
        # Step 4: Select only the features used by the model (after feature selection)
        df_final = df_scaled[self.feature_names]
        
        # Step 5: Predict
        proba = self.model.predict_proba(df_final)[0, 1]
        predicted_class = 1 if proba >= self.fraud_threshold else 0
        predicted_label = "illicit" if predicted_class == 1 else "licit"
        
        # Risk level
        if proba < self.risk_low:
            risk_level = "low"
        elif proba < self.risk_medium:
            risk_level = "medium"
        elif proba < self.risk_high:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "fraud_probability": float(proba),
            "predicted_class": int(predicted_class),
            "predicted_label": predicted_label,
            "risk_level": risk_level,
            "processing_time_ms": round(processing_time, 2)
        }
