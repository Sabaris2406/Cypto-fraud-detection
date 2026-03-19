# config.py
import os

class Config:
    """Base configuration class."""
    DEBUG      = False
    TESTING    = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "crypto-fraud-detection-2026")

    # Model artifact paths - use environment variable or default to current directory
    BASE_DIR = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH    = os.path.join(BASE_DIR, "crypto_fraud_xgboost.json")
    SCALER_PATH   = os.path.join(BASE_DIR, "scaler.pkl")
    FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")
    IMPUTER_PATH  = os.path.join(BASE_DIR, "imputer.pkl")
    FRAUD_THRESHOLD = 0.5

    RISK_LOW    = 0.30
    RISK_MEDIUM = 0.60
    RISK_HIGH   = 0.85

    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:5000",
                    "http://localhost:5000"]

    MAX_BATCH_SIZE = 100


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG           = False
    FRAUD_THRESHOLD = 0.45
