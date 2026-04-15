# BiasSpectra 

**ML-powered political bias detection for Indian news headlines**

🚀 **[Try the Live Demo](https://huggingface.co/spaces/dakshaditya/bias-spectra)**

BiasSpectra is an NLP system that detects ideological bias (**Left / Neutral / Right**) in Indian news headlines using a hybrid approach: rule-based keyword filtering + **zero-shot NLI classification** (DeBERTa). Unlike source-based training approaches, BiasSpectra actually *understands* the framing of each headline.

> ⚠️ This is a **media literacy and research tool**, not a fact-checking system. Predictions reflect *linguistic framing*, not factual correctness.

---

## How It Works

```
                    ┌─────────────────┐
                    │  Input Headline  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Gate 1: Is it   │──── Yes ──→ Neutral (Non-Political)
                    │  non-political?  │              weather, sports, etc.
                    └────────┬────────┘
                             │ No
                    ┌────────▼────────┐
                    │  Gate 2: Contains │──── No ──→ Neutral (Unbiased Political)
                    │  political bias? │
                    └────────┬────────┘
                             │ Yes
                    ┌────────▼─────────────────┐
                    │  Gate 3: Zero-Shot NLI    │
                    │  Tests 10 hypotheses      │──→ Left / Neutral / Right
                    │  about political framing  │    + confidence scores
                    └───────────────────────────┘
```

**Bias Classes:**
| Class | Description |
|-------|-------------|
| 🟥 **Left** | Critical / accountability-focused framing |
| 🟡 **Neutral** | Descriptive / informational framing |
| 🟦 **Right** | Supportive / pro-establishment framing |

---

## Quick Start

### Installation

```bash           
git clone https://github.com/Adityahatake/bias-spectra.git
cd bias-spectra
pip install -r requirements.txt
```

### Web App (FastAPI + HTML/CSS/JS)

```bash
python run.py app
# or directly:
uvicorn src.app:app --reload
```

### CLI

```bash
# Single prediction
python run.py predict "Opposition criticizes government on farm laws"
python run.py predict --model baseline "Supreme Court hears plea"

# Train models
python run.py train --model baseline
python run.py train --model bert

# Evaluate models
python run.py evaluate
python run.py evaluate --model bert
```

---

## Project Structure

```
bias-spectra/
├── config.py                    # Centralized configuration
├── run.py                       # CLI entry point
├── requirements.txt             # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── app.py                   # FastAPI backend & static server
│   ├── frontend/                # Custom HTML/CSS/JS UI
│   ├── political_filter.py      # Rule-based headline filter
│   │
│   ├── data/
│   │   ├── scraper.py           # News headline scraper
│   │   └── preprocessor.py      # Data cleaning & balancing
│   │
│   ├── training/
│   │   ├── baseline.py          # TF-IDF + LogReg trainer
│   │   └── bert_trainer.py      # BERT fine-tuning trainer
│   │
│   ├── evaluation/
│   │   └── evaluator.py         # Unified model evaluator
│   │
│   └── inference/
│       └── predictor.py         # BiasPredictor engine
│
├── data/
│   ├── raw/                     # Scraped headlines (gitignored)
│   └── processed/               # Cleaned dataset (gitignored)
│
├── models/                      # Trained model artifacts (gitignored)
│
├── model_card.md                # ML Model Card
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Primary Model | DeBERTa-v3 Zero-Shot NLI (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) |
| Legacy Models | Fine-tuned mBERT, TF-IDF + LogReg |
| Framework | PyTorch + HuggingFace Transformers |
| Web App | Vanilla HTML/CSS/JS + FastAPI |

---

## Methodology: Multi-Hypothesis NLI

Instead of training on source-labeled data (which bakes in bias), BiasSpectra uses **zero-shot Natural Language Inference**. For each headline, the model tests 10 hypotheses:

| Hypothesis | Signal |
|-----------|--------|
| "This text criticizes the government or ruling party" | Left |
| "This text demands accountability from people in power" | Left |
| "This text highlights failures or corruption of the establishment" | Left |
| "This text advocates for social justice or civil liberties" | Left |
| "This text supports the government or ruling party" | Right |
| "This text promotes nationalism or national pride" | Right |
| "This text praises government policies or achievements" | Right |
| "This text frames opposition or dissent negatively" | Right |
| "This text is a factual news report without opinion" | Neutral |
| "This text describes events objectively without taking sides" | Neutral |

Scores are averaged per class, normalized, and the top class wins — unless the margin is too thin, in which case the headline defaults to Neutral.

---

## Known Limitations

- **Headline-only**: Currently analyzes only headlines, not full articles.
- **English-centric**: NLI model works best on English; mixed-language headlines may reduce accuracy.
- **Framing ambiguity**: Some headlines are genuinely ambiguous; low-confidence predictions default to Neutral.

---

## Author

**Aditya Daksh**


---

## License

This project is for educational and research purposes.
