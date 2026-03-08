FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create required directories
RUN mkdir -p models data logs

# Generate synthetic training data and train the model
# This ensures model artefacts exist at image build time.
# Models are intentionally excluded from git (.gitignore), so we
# produce them here instead of expecting them to be present in the repo.
RUN python scripts/generate_sample_data.py --rows 10000 --fraud_rate 0.01 \
        --output data/creditcard.csv \
    && python scripts/train_model.py --model xgboost

# Expose Flask port
EXPOSE 5000

# Default command: start the API
CMD ["python", "src/api/app.py"]
