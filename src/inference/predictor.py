"""
BiasPredictor – Unified prediction engine.
==========================================
Primary model: Zero-Shot NLI using DeBERTa (no training data needed).
Fallback: Fine-tuned BERT or TF-IDF baseline (legacy).

The NLI approach tests multiple hypotheses about political framing
and aggregates scores for a balanced Left/Neutral/Right prediction.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    BASELINE_MODEL,
    BASELINE_TFIDF,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    LABEL_MAP,
    NLI_CONFIDENCE_THRESHOLD,
    NLI_HYPOTHESES,
    NLI_MODEL_NAME,
)
from src.political_filter import FilterResult, PoliticalFilter

logger = logging.getLogger(__name__)


@dataclass
class BiasResult:
    """Structured prediction result."""
    label: str                                          # "Left", "Neutral", "Right"
    confidence: dict = field(default_factory=dict)      # {label: probability}
    gate: str = ""                                      # which gate triggered
    reasoning: str = ""                                 # human-readable explanation

    @property
    def is_model_prediction(self) -> bool:
        return self.gate == "model"


class BiasPredictor:
    """
    End-to-end bias prediction: filter → model → result.

    model_type options:
      - "nli"      (default, recommended) – Zero-shot DeBERTa NLI
      - "bert"     (legacy) – Fine-tuned multilingual BERT
      - "baseline" (legacy) – TF-IDF + Logistic Regression

    Usage:
        predictor = BiasPredictor(model_type="nli")
        result = predictor.predict("Opposition criticizes govt on farm laws")
        print(result.label, result.confidence)
    """

    def __init__(self, model_type: str = "nli") -> None:
        self.filter = PoliticalFilter()
        self.model_type = model_type
        self._model = None
        self._tokenizer = None
        self._vectorizer = None
        self._nli_pipeline = None

    def _load_model(self) -> None:
        """Lazy-load the ML model on first prediction."""
        if self.model_type == "nli" and self._nli_pipeline is not None:
            return
        if self.model_type != "nli" and self._model is not None:
            return

        if self.model_type == "nli":
            logger.info("Loading NLI model: %s (this may take a moment)...", NLI_MODEL_NAME)
            self._nli_pipeline = pipeline(
                "zero-shot-classification",
                model=NLI_MODEL_NAME,
                device=-1,  # CPU; set to 0 for GPU
            )
            logger.info("NLI model loaded successfully.")

        elif self.model_type == "bert":
            model_path = self._resolve_bert_path()
            self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.eval()
            logger.info("Loaded BERT model from %s", model_path)

        else:  # baseline
            self._model = joblib.load(BASELINE_MODEL)
            self._vectorizer = joblib.load(BASELINE_TFIDF)
            logger.info("Loaded baseline model from %s", BASELINE_MODEL)

    def predict(self, headline: str) -> BiasResult:
        """
        Predict bias for a single headline.

        The headline passes through rule-based gates first:
          Gate 1: Non-political → Neutral
          Gate 2: Political but no bias keywords → Neutral
          Gate 3: ML model inference
        """
        gate_result = self.filter.classify(headline)

        if gate_result == FilterResult.NON_POLITICAL:
            return BiasResult(
                label="Neutral",
                confidence={"Left": 0.0, "Neutral": 1.0, "Right": 0.0},
                gate="non_political",
                reasoning="Non-political content (weather, sports, lifestyle, etc.)",
            )

        if gate_result == FilterResult.NEUTRAL_POLITICAL:
            return BiasResult(
                label="Neutral",
                confidence={"Left": 0.0, "Neutral": 1.0, "Right": 0.0},
                gate="neutral_political",
                reasoning="Political topic with no ideological framing detected.",
            )

        # Gate 3: ML model
        self._load_model()

        if self.model_type == "nli":
            return self._predict_nli(headline)
        elif self.model_type == "bert":
            return self._predict_bert(headline)
        return self._predict_baseline(headline)

    # ── NLI Zero-Shot (Primary) ──────────────────────────────

    def _predict_nli(self, headline: str) -> BiasResult:
        """
        Multi-hypothesis NLI classification.

        For each bias class, we test multiple hypotheses and average
        the entailment scores. The class with the highest aggregated
        score wins — unless the gap is too small, in which case we
        default to Neutral (ambiguous framing).
        """
        # Collect all hypotheses and track which class they belong to
        all_hypotheses = []
        hypothesis_to_class = {}
        for cls, hyps in NLI_HYPOTHESES.items():
            for h in hyps:
                all_hypotheses.append(h)
                hypothesis_to_class[h] = cls

        # Run zero-shot on all hypotheses at once
        result = self._nli_pipeline(
            headline,
            candidate_labels=all_hypotheses,
            multi_label=True,  # each hypothesis scored independently
        )

        # Aggregate scores per class (average of hypothesis scores)
        class_scores = {"Left": 0.0, "Neutral": 0.0, "Right": 0.0}
        class_counts = {"Left": 0, "Neutral": 0, "Right": 0}

        for label, score in zip(result["labels"], result["scores"]):
            cls = hypothesis_to_class[label]
            class_scores[cls] += score
            class_counts[cls] += 1

        # Average per class
        for cls in class_scores:
            if class_counts[cls] > 0:
                class_scores[cls] /= class_counts[cls]

        # Normalize to sum to 1
        total = sum(class_scores.values())
        if total > 0:
            confidence = {cls: round(s / total, 4) for cls, s in class_scores.items()}
        else:
            confidence = {"Left": 0.33, "Neutral": 0.34, "Right": 0.33}

        # Determine prediction
        sorted_classes = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1]

        # If confidence gap is too small → Neutral (ambiguous)
        if (top_score - second_score) < NLI_CONFIDENCE_THRESHOLD:
            label = "Neutral"
            reasoning = (
                f"NLI analysis: ambiguous framing "
                f"(gap {top_score - second_score:.1%} < threshold {NLI_CONFIDENCE_THRESHOLD:.0%})"
            )
        else:
            label = top_label
            reasoning = f"NLI analysis: {label} framing detected ({confidence[label]:.1%} confidence)"

        return BiasResult(
            label=label,
            confidence=confidence,
            gate="model",
            reasoning=reasoning,
        )

    # ── BERT (Legacy) ────────────────────────────────────────

    def _predict_bert(self, headline: str) -> BiasResult:
        with torch.no_grad():
            inputs = self._tokenizer(
                headline, return_tensors="pt",
                truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
            )
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        confidence = {LABEL_MAP[i]: round(probs[i].item(), 4) for i in LABEL_MAP}
        return BiasResult(
            label=LABEL_MAP[pred_idx],
            confidence=confidence,
            gate="model",
            reasoning=f"BERT model prediction ({confidence[LABEL_MAP[pred_idx]]:.1%} confidence)",
        )

    # ── Baseline (Legacy) ────────────────────────────────────

    def _predict_baseline(self, headline: str) -> BiasResult:
        vec = self._vectorizer.transform([headline])
        pred = self._model.predict(vec)[0]
        proba = self._model.predict_proba(vec)[0]
        classes = self._model.classes_

        confidence = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        for lbl in ("Left", "Neutral", "Right"):
            confidence.setdefault(lbl, 0.0)

        return BiasResult(
            label=pred,
            confidence=confidence,
            gate="model",
            reasoning=f"Baseline model prediction ({confidence[pred]:.1%} confidence)",
        )

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve_bert_path() -> str:
        bert_dir = Path(BERT_MODEL_DIR)
        if (bert_dir / "config.json").exists():
            return str(bert_dir)
        checkpoints = sorted(bert_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            return str(checkpoints[-1])
        raise FileNotFoundError(f"No BERT model found in {bert_dir}")
