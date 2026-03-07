# Dataset

## Option 1 — Real Dataset (Kaggle)

1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in this `data/` folder
3. The file should have 284,807 rows and 31 columns:
   - `Time` — seconds since first transaction
   - `V1` – `V28` — PCA-transformed features (anonymized)
   - `Amount` — transaction amount in EUR
   - `Class` — 0 = legit, 1 = fraud

## Option 2 — Synthetic Dataset (for testing)

Generate a synthetic dataset (no Kaggle account needed):

```bash
python scripts/generate_sample_data.py --rows 50000 --output data/creditcard.csv
```

Then update `config.yaml`:
```yaml
data:
  raw_path: data/creditcard.csv
```

## Dataset Statistics (Real)

| Metric | Value |
|--------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Legitimate | 284,315 (99.83%) |
| Features | 30 |
| Time span | 2 days |
