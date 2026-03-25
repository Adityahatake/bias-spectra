"""
BaselineTrainer – TF-IDF + Logistic Regression trainer.
=======================================================
Consolidated replacement for train_baseline.py and train_3class.py.
Supports configurable n-grams, class weighting, and saves
model artifacts alongside a classification report.
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    BASELINE_MODEL,
    BASELINE_TFIDF,
    MODELS_DIR,
    PROCESSED_CSV,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    map_5_to_3,
)

logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Train a TF-IDF + Logistic Regression model for 3-class bias detection.

    Usage:
        trainer = BaselineTrainer()
        trainer.train()
    """

    def __init__(
        self,
        data_path=PROCESSED_CSV,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
        test_size: float = 0.2,
        class_weight: str = "balanced",
    ) -> None:
        self.data_path = data_path
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.class_weight = class_weight

    def train(self) -> dict:
        """Full training pipeline. Returns classification report dict."""
        # Load & prepare data
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["clean_headline", "category"])
        df = df[df["clean_headline"].str.strip() != ""]

        # Use 3-class labels
        df["bias"] = df["category"].apply(map_5_to_3)

        X = df["clean_headline"].astype(str)
        y = df["bias"]

        logger.info("Class distribution:\n%s", y.value_counts().to_string())

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train
        model = LogisticRegression(max_iter=3000, class_weight=self.class_weight)
        model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logger.info("\n==== CLASSIFICATION REPORT ====\n%s", report_str)
        logger.info("\n==== CONFUSION MATRIX ====\n%s", cm)

        # Save artifacts
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, BASELINE_MODEL)
        joblib.dump(vectorizer, BASELINE_TFIDF)

        report_path = MODELS_DIR / "baseline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Model  → %s", BASELINE_MODEL)
        logger.info("TF-IDF → %s", BASELINE_TFIDF)
        logger.info("Report → %s", report_path)

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    trainer = BaselineTrainer()
    trainer.train()
