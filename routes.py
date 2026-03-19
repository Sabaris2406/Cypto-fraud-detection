"""
API routes for cryptocurrency fraud detection
"""
from flask import Blueprint, request, jsonify, current_app
from validators import validate_predict_request, validate_batch_request
import logging

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "XGBoost Crypto Fraud Detector v1.0",
        "version": "1.0.0",
    }), 200


@api_bp.route("/model/info", methods=["GET"])
def model_info():
    """Get model metadata"""
    cfg = current_app.config
    return jsonify({
        "model_type": "XGBoost Classifier",
        "dataset": "Elliptic Bitcoin Dataset",
        "total_features": 103,
        "fraud_threshold": cfg.get("FRAUD_THRESHOLD", 0.5),
        "expected_metrics": {
            "precision": ">0.99",
            "recall": ">0.94",
            "f1_score": ">0.96",
            "auc_roc": ">0.99",
        },
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Single transaction prediction"""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    is_valid, err = validate_predict_request(data)
    if not is_valid:
        return jsonify({"error": err}), 400

    try:
        predictor = current_app.config["PREDICTOR"]
        result = predictor.predict(data)
        logger.info(
            f"Prediction: {result['predicted_label']} "
            f"prob={result['fraud_probability']:.4f}"
        )
        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error during prediction"}), 500


@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch transaction prediction"""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON in request body"}), 400
    
    max_batch = current_app.config.get("MAX_BATCH_SIZE", 100)
    is_valid, err = validate_batch_request(data, max_batch)
    if not is_valid:
        return jsonify({"error": err}), 400

    transactions = data["transactions"]

    try:
        predictor = current_app.config["PREDICTOR"]
        results = []
        for i, txn in enumerate(transactions):
            ok, verr = validate_predict_request(txn)
            if not ok:
                results.append({"index": i, "error": verr})
            else:
                results.append({"index": i, "result": predictor.predict(txn)})

        illicit_count = sum(
            1 for r in results
            if "result" in r and r["result"]["predicted_label"] == "illicit"
        )
        return jsonify({
            "status": "success",
            "total": len(transactions),
            "illicit_flagged": illicit_count,
            "results": results,
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": "Internal server error during batch prediction"}), 500
