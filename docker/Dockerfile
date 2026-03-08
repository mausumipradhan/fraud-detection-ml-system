FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Tell Python where the project root is
ENV PYTHONPATH=/app

# Create dummy model so API can start in CI
RUN mkdir -p models && python - <<EOF
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

model = DummyClassifier()
model.fit(np.zeros((10,29)), np.zeros(10))

joblib.dump(model, "models/xgboost_model.joblib")
EOF

# Expose API port
EXPOSE 5000

# Start API
CMD ["python", "src/api/app.py"]