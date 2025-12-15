import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/indicbert_bias/checkpoint-124"

MODEL_NAME = "bert-base-multilingual-cased"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load test data
df = pd.read_csv("data/processed/india_clean_dataset.csv")

def map_label(label):
    if label in ["Left", "Left-Center"]:
        return 0
    elif label == "Center":
        return 1
    else:
        return 2

df = df.dropna(subset=["clean_headline", "category"])
df["label"] = df["category"].apply(map_label)

# Use same split logic
test_df = df.sample(frac=0.2, random_state=42)

X = test_df["clean_headline"].tolist()
y_true = test_df["label"].tolist()

# Predict
y_pred = []

with torch.no_grad():
    for text in X:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        y_pred.append(pred)

# Report
print("\n==== BERT CLASSIFICATION REPORT ====")
print(classification_report(y_true, y_pred, target_names=["Left", "Neutral", "Right"]))

print("\n==== CONFUSION MATRIX ====")
print(confusion_matrix(y_true, y_pred))
