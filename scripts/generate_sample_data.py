#!/usr/bin/env python
"""
scripts/generate_sample_data.py
Generate synthetic credit card transaction data for testing.

Usage:
    python scripts/generate_sample_data.py --rows 10000 --fraud_rate 0.002
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(
    n_rows: int = 10000, fraud_rate: float = 0.002, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # PCA-style features V1–V28
    legit_features = rng.standard_normal((n_legit, 28))
    fraud_features = rng.standard_normal((n_fraud, 28)) * 1.5 + 0.5  # shifted

    legit_amount = rng.exponential(scale=100, size=n_legit)
    fraud_amount = rng.exponential(scale=300, size=n_fraud)

    legit_time = np.sort(rng.uniform(0, 172800, n_legit))
    fraud_time = rng.uniform(0, 172800, n_fraud)

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    legit_df = pd.DataFrame(
        np.column_stack([legit_time, legit_features, legit_amount, np.zeros(n_legit)]),
        columns=cols,
    )
    fraud_df = pd.DataFrame(
        np.column_stack([fraud_time, fraud_features, fraud_amount, np.ones(n_fraud)]),
        columns=cols,
    )

    df = (
        pd.concat([legit_df, fraud_df])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    df["Class"] = df["Class"].astype(int)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--fraud_rate", type=float, default=0.002)
    parser.add_argument("--output", default="data/creditcard_synthetic.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate(n_rows=args.rows, fraud_rate=args.fraud_rate)
    df.to_csv(args.output, index=False)

    print(f"✅ Generated {len(df):,} rows → {args.output}")
    print(f"   Legit: {(df['Class']==0).sum():,}  Fraud: {(df['Class']==1).sum():,}")
