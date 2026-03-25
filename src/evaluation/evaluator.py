"""
ModelEvaluator – Unified evaluation for both Baseline and BERT models.
=====================================================================
Generates classification reports, confusion matrices, and saves
results to JSON for easy comparison.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    BASELINE_MODEL,
    BASELINE_TFIDF,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    LABEL_MAP,
    LABEL_MAP_INV,
    MODELS_DIR,
    PROCESSED_CSV,
    map_5_to_3,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance on the processed dataset.

    Supports both sklearn baseline models and BERT models.

    Usage:
        evaluator = ModelEvaluator()
        evaluator.evaluate_baseline()
        evaluator.evaluate_bert()
    """

    def __init__(self, data_path=PROCESSED_CSV) -> None:
        self.data_path = data_path

    def evaluate_baseline(self) -> dict:
        """Evaluate TF-IDF + Logistic Regression model."""
        logger.info("Evaluating baseline model...")
        df = self._load_data()

        model = joblib.load(BASELINE_MODEL)
        vectorizer = joblib.load(BASELINE_TFIDF)

        X_test = df["clean_headline"].astype(str)
        y_test = df["bias"]

        X_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_vec)

        return self._report("baseline", y_test, y_pred)

    def evaluate_bert(self, checkpoint: str | None = None) -> dict:
        """Evaluate BERT model. Auto-detects latest checkpoint if not specified."""
        logger.info("Evaluating BERT model...")
        df = self._load_data()

        model_path = self._resolve_bert_path(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        X_test = df["clean_headline"].tolist()
        y_true = df["bias"].apply(lambda x: LABEL_MAP_INV[x]).tolist()

        y_pred = []
        with torch.no_grad():
            for text in X_test:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    padding=True, max_length=64,
                )
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                y_pred.append(pred)

        y_true_labels = [LABEL_MAP[i] for i in y_true]
        y_pred_labels = [LABEL_MAP[i] for i in y_pred]

        return self._report("bert", y_true_labels, y_pred_labels)

    # ── Helpers ──────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["clean_headline", "category"])
        df["bias"] = df["category"].apply(map_5_to_3)
        # Use 20% as test set (same split seed as training)
        return df.sample(frac=0.2, random_state=42)

    @staticmethod
    def _resolve_bert_path(checkpoint: str | None) -> str:
        if checkpoint:
            return checkpoint
        # Auto-detect: look for checkpoint dirs or use the base dir
        bert_dir = Path(BERT_MODEL_DIR)
        checkpoints = sorted(bert_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            return str(checkpoints[-1])
        # If model files exist directly in BERT_MODEL_DIR
        if (bert_dir / "config.json").exists():
            return str(bert_dir)
        raise FileNotFoundError(f"No BERT model found in {bert_dir}")

    @staticmethod
    def _report(model_name: str, y_true, y_pred) -> dict:
        report = classification_report(y_true, y_pred, output_dict=True)
        report_str = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        logger.info("\n==== %s CLASSIFICATION REPORT ====\n%s", model_name.upper(), report_str)
        logger.info("\n==== CONFUSION MATRIX ====\n%s", cm)

        # Save
        out_path = MODELS_DIR / f"{model_name}_eval_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved → %s", out_path)

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    evaluator = ModelEvaluator()

    try:
        evaluator.evaluate_baseline()
    except FileNotFoundError:
        logger.warning("Baseline model not found – skipping.")

    try:
        evaluator.evaluate_bert()
    except FileNotFoundError:
        logger.warning("BERT model not found – skipping.")
