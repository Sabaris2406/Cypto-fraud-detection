import logging
import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from config import DevelopmentConfig, ProductionConfig
from routes import api_bp
from predictor import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    for attr in ["FRAUD_THRESHOLD", "RISK_LOW", "RISK_MEDIUM", "RISK_HIGH", "MAX_BATCH_SIZE"]:
        app.config[attr] = getattr(config_class, attr)
    
    CORS(app, origins=config_class.CORS_ORIGINS)
    
    logger.info("Loading fraud detection model...")
    predictor = FraudPredictor(
        model_path=config_class.MODEL_PATH,
        scaler_path=config_class.SCALER_PATH,
        features_path=config_class.FEATURES_PATH,
        imputer_path=config_class.IMPUTER_PATH,
        config=app.config,
    )
    app.config["PREDICTOR"] = predictor
    logger.info("Model loaded successfully")
    
    app.register_blueprint(api_bp)
    
    @app.route("/")
    def index():
        return render_template("index.html")
    
    return app

if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=True)

# For production (Render, Heroku, etc.)
application = create_app()
