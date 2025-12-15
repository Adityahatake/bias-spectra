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

## ⚠️ Known Limitation: Left-Leaning Bias

The current model shows a mild Left-leaning bias in low-confidence cases.

**Reason:**
- Indian news language contains richer linguistic patterns for criticism
  than for explicit pro-government advocacy.
- This results in uncertainty cases leaning toward Left due to learned priors.

This behavior reflects real-world media patterns and is **documented intentionally**.

---

## 🚀 Planned Improvements

- Dataset rebalancing with class-weighted loss
- Confidence-threshold-based Neutral fallback
- Right-framing calibration
- Full-article (not just headline) analysis
- Improved evaluation using macro-F1

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

## 📂 Project Structure

