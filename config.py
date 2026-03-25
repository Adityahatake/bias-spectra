"""
BiasSpectra – Centralized Configuration
========================================
All paths, hyperparameters, label maps, and scraper settings live here.
Every module imports from this file instead of hardcoding values.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = ROOT_DIR / "models"

RAW_CSV        = RAW_DATA_DIR / "india_news_raw.csv"
PROCESSED_CSV  = PROCESSED_DIR / "india_clean_dataset.csv"

BASELINE_MODEL = MODELS_DIR / "bias_model_3class.pkl"
BASELINE_TFIDF = MODELS_DIR / "tfidf_vectorizer_3class.pkl"
BERT_MODEL_DIR = MODELS_DIR / "indicbert_bias"

# ── Label Schema ──────────────────────────────────────────────
LABEL_MAP = {0: "Left", 1: "Neutral", 2: "Right"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

def map_5_to_3(label: str) -> str:
    """Collapse the 5-class source taxonomy into 3 classes."""
    if label in ("Left", "Left-Center"):
        return "Left"
    if label == "Center":
        return "Neutral"
    return "Right"  # Center-Right, Right

# ── NLI Zero-Shot Settings (PRIMARY MODEL) ────────────────────
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Each hypothesis is tested against the headline via NLI entailment.
# Scores are aggregated per bias class for the final prediction.
NLI_HYPOTHESES = {
    "Left": [
        "This text criticizes the government or ruling party.",
        "This text demands accountability from people in power.",
        "This text highlights failures or corruption of the establishment.",
        "This text advocates for social justice, minority rights, or civil liberties.",
    ],
    "Right": [
        "This text supports the government or ruling party.",
        "This text promotes nationalism, national pride, or cultural heritage.",
        "This text praises government policies or achievements.",
        "This text frames opposition or dissent negatively.",
    ],
    "Neutral": [
        "This text is a factual news report without opinion or bias.",
        "This text describes events objectively without taking sides.",
    ],
}

# Minimum confidence gap between top-2 classes to make a call;
# below this threshold the headline is classified Neutral.
NLI_CONFIDENCE_THRESHOLD = 0.08

# ── BERT Settings (LEGACY) ───────────────────────────────────
BERT_MODEL_NAME   = "bert-base-multilingual-cased"
BERT_MAX_LENGTH   = 64
BERT_TRAIN_EPOCHS = 3
BERT_BATCH_SIZE   = 16
BERT_LEARNING_RATE = 2e-5

# ── Baseline Settings ────────────────────────────────────────
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE  = (1, 3)

# ── Scraper Settings ─────────────────────────────────────────
SCRAPE_TARGET_TOTAL    = 1500
SCRAPE_MAX_PAGES       = 30
SCRAPE_DELAY_RANGE     = (0.6, 1.5)
SCRAPE_USER_AGENT      = (
    "Mozilla/5.0 (compatible; BiasBot/1.0; +https://github.com/Adityahatake/bias-spectra)"
)

SCRAPE_SOURCES = {
    "thewire.in": {
        "base": "https://thewire.in",
        "category": "Left",
        "sections": ["/politics", "/science-technology", "/society"],
    },
    "scroll.in": {
        "base": "https://scroll.in",
        "category": "Left",
        "sections": ["/topic/politics", "/topic/social-issues"],
    },
    "thenewsminute.com": {
        "base": "https://www.thenewsminute.com",
        "category": "Left",
        "sections": ["/categories/politics", "/categories/national"],
    },
    "caravanmagazine.in": {
        "base": "https://caravanmagazine.in",
        "category": "Left",
        "sections": ["/", "/politics"],
    },
    "thehindu.com": {
        "base": "https://www.thehindu.com",
        "category": "Left-Center",
        "sections": ["/news", "/news/national"],
    },
    "indianexpress.com": {
        "base": "https://indianexpress.com",
        "category": "Left-Center",
        "sections": ["/section/india", "/section/politics"],
    },
    "indiatoday.in": {
        "base": "https://www.indiatoday.in",
        "category": "Center",
        "sections": ["/india", "/politics"],
    },
    "aajtak.in": {
        "base": "https://www.aajtak.in",
        "category": "Center",
        "sections": ["/english", "/news"],
    },
    "ndtv.com": {
        "base": "https://www.ndtv.com",
        "category": "Center-Right",
        "sections": ["/latest", "/india", "/politics"],
    },
    "economictimes.indiatimes.com": {
        "base": "https://economictimes.indiatimes.com",
        "category": "Center-Right",
        "sections": ["/policy", "/news/india"],
    },
    "timesnownews.com": {
        "base": "https://www.timesnownews.com",
        "category": "Right",
        "sections": ["/india", "/politics"],
    },
    "republicworld.com": {
        "base": "https://www.republicworld.com",
        "category": "Right",
        "sections": ["/national", "/politics"],
    },
    "zeenews.india.com": {
        "base": "https://zeenews.india.com",
        "category": "Right",
        "sections": ["/india", "/politics"],
    },
    "news18.com": {
        "base": "https://www.news18.com",
        "category": "Right",
        "sections": ["/news/india", "/news/politics"],
    },
}
