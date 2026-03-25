"""
BertTrainer – Fine-tune BERT for political bias classification.
===============================================================
Config-driven trainer using HuggingFace Transformers with
integrated evaluation metrics and proper train/eval split.
"""

import logging

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    BERT_BATCH_SIZE,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_TRAIN_EPOCHS,
    LABEL_MAP,
    PROCESSED_CSV,
    map_5_to_3,
    LABEL_MAP_INV,
)

logger = logging.getLogger(__name__)


class BertTrainer:
    """
    Fine-tune a multilingual BERT model for 3-class bias detection.

    Usage:
        trainer = BertTrainer()
        trainer.train()
    """

    def __init__(
        self,
        data_path=PROCESSED_CSV,
        model_name: str = BERT_MODEL_NAME,
        output_dir=BERT_MODEL_DIR,
        epochs: int = BERT_TRAIN_EPOCHS,
        batch_size: int = BERT_BATCH_SIZE,
        lr: float = BERT_LEARNING_RATE,
        max_length: int = BERT_MAX_LENGTH,
    ) -> None:
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = str(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self) -> None:
        """Full training pipeline with evaluation."""
        # Prepare data
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["clean_headline", "category"])
        df = df[df["clean_headline"].str.strip() != ""]

        df["label"] = df["category"].apply(
            lambda x: LABEL_MAP_INV[map_5_to_3(x)]
        )

        logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

        dataset = Dataset.from_pandas(df[["clean_headline", "label"]])
        dataset = dataset.map(self._tokenize, batched=True)
        split = dataset.train_test_split(test_size=0.2, seed=42)

        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(LABEL_MAP)
        )

        # Training arguments
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting BERT fine-tuning (%d epochs)", self.epochs)
        trainer.train()

        # Save final model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Model saved → %s", self.output_dir)

    def _tokenize(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["clean_headline"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    @staticmethod
    def _compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    trainer = BertTrainer()
    trainer.train()
