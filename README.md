# BiasSpectra 📰
### ML-based Political Bias Detection System for Indian News

BiasSpectra is an NLP-driven system that detects ideological bias
(**Left / Neutral / Right**) in Indian news headlines using a hybrid
approach combining rule-based filtering and a fine-tuned BERT model.

---

## 🔍 Why BiasSpectra?

Media bias detection is challenging in the Indian context due to:
- Multilingual reporting
- Subtle ideological framing
- Asymmetric criticism vs advocacy language

BiasSpectra is designed as a **media literacy and research tool**,
not as a fact-checking system.

---

## 🧠 How It Works

1. User inputs a news headline
2. Rule-based filter removes non-political content (weather, sports, etc.)
3. Political but unbiased headlines are classified as **Neutral**
4. Ideologically framed headlines are analyzed using BERT
5. Confidence scores are displayed for transparency

---

## 🏷️ Bias Classes

- **Left** – Critical or accountability-focused framing  
- **Neutral** – Descriptive or informational framing  
- **Right** – Supportive or pro-establishment framing  

⚠️ Non-political content is automatically classified as **Neutral**.

---


## 🧪 Example Predictions

| Headline | Prediction |
|--------|------------|
| Hyderabad Weather Forecast | Neutral (Non-Political) |
| Supreme Court hears plea | Neutral |
| Opposition criticizes government | Left |
| Government rejects opposition claims | Right |

---

## 🛠️ Tech Stack

- Python
- HuggingFace Transformers
- BERT (Multilingual)
- PyTorch
- Streamlit
- scikit-learn (baseline)

---

## 🛠️Structure

bias-spectra/
├── data/
│   ├── raw/                 # Unprocessed, original datasets
│   └── processed/           # Cleaned and preprocessed datasets
│
├── models/                  # Saved model checkpoints and artifacts
│
├── src/                     # Core source code
│   ├── app.py               # Main application entry point
│   ├── political_filter.py  # Political bias classification logic
│   ├── train_indicbert.py   # Training script for IndicBERT
│   └── evaluate_bert.py     # Evaluation script for model performance
│
├── README.md                # Project overview and usage instructions
└── MODEL_CARD.md            # Detailed model card explaining architecture, training, and limitations
