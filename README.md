# Cryptocurrency Fraud Detection System

AI-powered Bitcoin transaction fraud detection using XGBoost.

## Features
- 99% accuracy fraud detection
- Real-time predictions (< 10ms)
- REST API + Web Interface
- Batch processing support

## Model Performance
- Precision: 99%
- Recall: 94%
- AUC-ROC: 0.9985

## API Endpoints
- `GET /api/health` - Health check
- `GET /api/model/info` - Model metadata
- `POST /api/predict` - Single prediction
- `POST /api/predict/batch` - Batch predictions

## Local Setup
```bash
pip install -r requirements.txt
python app.py
```

## Project Info
- Author: Krishna Kumar S (23CTU033)
- Guide: Mrs. R. Poongodi
- Dataset: Elliptic Bitcoin Dataset
