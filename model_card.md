# Model Card: BiasSpectra

## Model Details

| Field | Value |
|-------|-------|
| **Name** | BiasSpectra Political Bias Classifier |
| **Primary Model** | Zero-Shot NLI via `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| **Legacy Models** | Fine-tuned `bert-base-multilingual-cased`, TF-IDF + LogReg |
| **Task** | 3-class text classification (Left / Neutral / Right) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Author** | Aditya Daksh |

---

## Intended Use

- **Primary**: Research and educational tool for understanding media framing patterns in Indian news
- **Users**: Students, researchers, journalists studying media bias
- **NOT intended for**: Fact-checking, censorship, content moderation, or automated editorial decisions

---

## Approach: Zero-Shot Natural Language Inference

BiasSpectra uses **zero-shot NLI classification** instead of supervised training on source-labeled data. This eliminates the source-based labeling bias that plagued earlier versions.

### How It Works

1. The headline is tested against **10 hypothesis statements** via NLI entailment
2. Each hypothesis probes a specific framing dimension (criticism, accountability, support, nationalism, factual reporting, etc.)
3. Entailment scores are **averaged per bias class** and normalized
4. The class with the highest score wins — unless the margin is too thin, triggering a Neutral fallback

### Why Not Source-Based Training?

Source-based labeling assumes every headline from a "left-leaning" outlet is biased left. This is false — most headlines are factual reports. The NLI approach evaluates each headline on its own linguistic content.

---

## Evaluation

The model uses multi-hypothesis NLI scoring. Evaluation metrics:
- Per-hypothesis entailment accuracy
- Class distribution balance across test headlines
- Confidence calibration (gap between top-2 classes)

---

## Limitations & Biases

1. **English-centric** — DeBERTa works best on English text; Hindi/mixed-language headlines may reduce accuracy
2. **Headline-only** — Full article analysis would improve framing detection
3. **Hypothesis design** — The 10 hypotheses capture common Indian political framing but may miss edge cases
4. **Ambiguous headlines** — Low-confidence cases default to Neutral rather than making an uncertain call

---

## Ethical Considerations

- Predictions reflect **linguistic framing patterns**, not factual accuracy
- Model outputs should **never** be used to suppress content or make editorial judgments
- The Left/Neutral/Right taxonomy is a simplification of a complex political spectrum
- Designed for **transparency and media literacy**, not judgment

---

## Author

**Aditya Daksh**
BTech CSE (AI)
