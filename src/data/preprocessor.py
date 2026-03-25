"""
DataPreprocessor – Clean and balance scraped headline data.
===========================================================
Handles text cleaning, 5→3 class label mapping, class balancing,
and deduplication. Config-driven paths and targets.
"""

import logging
import math
import re

import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    PROCESSED_CSV,
    PROCESSED_DIR,
    RAW_CSV,
    SCRAPE_TARGET_TOTAL,
    map_5_to_3,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clean raw scraped data and produce a balanced 3-class dataset.

    Usage:
        preprocessor = DataPreprocessor()
        df = preprocessor.run()  # returns DataFrame and saves CSV
    """

    def __init__(
        self,
        raw_csv=RAW_CSV,
        out_csv=PROCESSED_CSV,
        target_total: int = SCRAPE_TARGET_TOTAL,
    ) -> None:
        self.raw_csv = raw_csv
        self.out_csv = out_csv
        self.target_total = target_total

    # ── Public ───────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Full pipeline: load → clean → map labels → balance → save."""
        df = pd.read_csv(self.raw_csv)
        logger.info("Raw rows loaded: %d", len(df))

        df = df[df["headline"].notna()].copy()
        df["clean_headline"] = df["headline"].apply(self.clean_text)

        # Map original 5-class to 3-class
        df["category"] = df["category"].str.strip()
        df["bias"] = df["category"].apply(map_5_to_3)

        logger.info("Distribution before balancing: %s", df["bias"].value_counts().to_dict())

        df = self._balance(df)
        df = df.drop_duplicates(subset=["clean_headline"])
        df = df[["headline", "clean_headline", "url", "source", "category", "bias"]]

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)
        logger.info("Saved → %s (%d rows)", self.out_csv, len(df))
        logger.info("Final distribution: %s", df["bias"].value_counts().to_dict())
        return df

    # ── Text cleaning ────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize a headline for model input."""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)          # remove URLs
        text = re.sub(r"[^a-z0-9\s]", " ", text)      # keep only alphanumeric
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Balancing ────────────────────────────────────────────

    def _balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equalize class sizes via undersampling + optional overflow fill."""
        classes = sorted(df["bias"].unique())
        per_class = math.ceil(self.target_total / len(classes))
        logger.info("Balancing → %d per class (%s)", per_class, classes)

        parts = []
        for cls in classes:
            sub = df[df["bias"] == cls]
            sample = sub.sample(n=min(per_class, len(sub)), random_state=42)
            parts.append(sample)

        balanced = pd.concat(parts, ignore_index=True)

        # Fill shortage from remaining rows
        if len(balanced) < self.target_total:
            needed = self.target_total - len(balanced)
            remaining = df[~df.index.isin(balanced.index)]
            if len(remaining) > 0:
                fill = remaining.sample(n=min(needed, len(remaining)), random_state=42)
                balanced = pd.concat([balanced, fill], ignore_index=True)

        return balanced


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    preprocessor = DataPreprocessor()
    preprocessor.run()
