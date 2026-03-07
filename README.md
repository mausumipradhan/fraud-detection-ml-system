# Real-Time Credit Card Fraud Detection System

**Author:** Mausumi Pradhan  


A **production-ready machine learning system** for detecting fraudulent credit card transactions in real time.  
The system exposes a REST API for predictions, provides explainability using SHAP, includes monitoring utilities, and can be deployed using Docker.

---

# Model Performance

| Metric | Score |
|------|------|
| ROC-AUC | **0.99** |
| Precision | **0.95** |
| Recall | **0.92** |
| F1 Score | **0.93** |
| Best Model | **XGBoost Classifier** |

The model was trained on the **Kaggle Credit Card Fraud Detection dataset** and optimized for high recall while maintaining strong precision.

---

# Tech Stack

### Machine Learning
- Python
- XGBoost
- Scikit-learn
- SHAP (Explainable AI)

### Backend
- Flask REST API
- Pydantic validation

### Deployment
- Docker
- Docker Compose

### Visualization
- Streamlit dashboard

### Testing
- PyTest

---

# System Architecture

```
Transaction JSON
       │
       ▼
┌─────────────────┐
│   Flask API     │
│   (/predict)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Input Validator │
│   (Pydantic)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ StandardScaler  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XGBoost Model  │
│ Fraud Prediction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SHAP Explainer  │
│ Feature Impact  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ JSON Response   │
└─────────────────┘
```

---

# Project Structure

```
fraud-detection/
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── preprocessing/
│   │   ├── pipeline.py
│   │   └── validator.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   └── monitoring/
│       └── drift_detector.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/
│   ├── train_model.py
│   └── generate_sample_data.py
│
├── tests/
│
├── data/
│
├── requirements.txt
├── config.yaml
└── setup.py
```

---

# Quick Start

##  Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file in:

```
data/creditcard.csv
```

---

## Train the Model

```bash
python scripts/train_model.py --model xgboost --output models/
```

---

## Run the API

```bash
python src/api/app.py
```

API will start at:

```
http://localhost:5000
```

---

# Run with Docker

Build the Docker image:

```bash
docker build -t fraud-api -f docker/Dockerfile .
```

Run the container:

```bash
docker run -p 5000:5000 fraud-api
```

API will be available at:

```
http://localhost:5000
```

---

# Run with Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:

| Service | URL |
|------|------|
| API | http://localhost:5000 |
| Dashboard | http://localhost:8501 |

---

# API Reference

## POST `/predict`

Predict whether a transaction is fraudulent.

### Request

```json
{
  "features": [
    0.0, -1.35, -0.07, 2.53, 1.37, -0.33,
    0.46, 0.23, 0.09, 0.36, 0.09, -0.55,
    -0.61, -0.99, -0.31, 1.46, -0.47, 0.20,
    0.02, 0.40, 0.25, -0.01, 0.27, -0.11,
    0.06, -0.08, -0.25, -0.31
  ],
  "amount": 149.62
}
```

**features → 28 PCA features (V1–V28)**  
**amount → transaction amount**

---

### Response

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0023,
  "confidence": "HIGH",
  "explanation": {
    "top_features": [
      {"feature": "V14", "impact": -0.82},
      {"feature": "V4", "impact": 0.31}
    ]
  },
  "timestamp": "2026-03-03T20:53:20Z"
}
```

---

## GET `/health`

Health check endpoint.

---

## GET `/metrics`

Returns model performance metrics and drift status.

---

# Streamlit Monitoring Dashboard

Run the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard features:

- Live transaction monitoring
- Fraud rate visualization
- Feature importance charts
- Model drift alerts

---

# Testing

Run unit tests:

```bash
pytest tests/ -v --cov=src
```

---

# Configuration

Edit `config.yaml` to customize settings.

Example:

```yaml
model:
  type: xgboost
  threshold: 0.5

api:
  host: 0.0.0.0
  port: 5000

monitoring:
  drift_threshold: 0.05
```

---

# 🔮 Future Improvements

- Deep learning models (LSTM / Autoencoder)
- Real-time streaming with Kafka
- Advanced drift detection
- Automated model retraining
- CI/CD deployment pipeline

---

# License

MIT License — see `LICENSE` file.

---

# Project Highlights

✔ End-to-end ML pipeline  
✔ Explainable AI using SHAP  
✔ REST API for real-time predictions  
✔ Dockerized deployment  
✔ Monitoring dashboard  