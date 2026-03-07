#!/usr/bin/env python
"""
scripts/train_model.py
CLI wrapper for the training pipeline.

Usage:
    python scripts/train_model.py --model xgboost
    python scripts/train_model.py --model random_forest --config config.yaml
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--model",
        choices=["xgboost", "random_forest", "logistic_regression"],
        default="xgboost",
        help="Model type to train (default: xgboost)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n🚀 Training {args.model.upper()} model...\n")
    metrics = train(config_path=args.config, model_type=args.model)
    print("\n✅ Training complete!")
    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
