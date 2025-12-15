# src/train_3class.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/processed/india_clean_dataset.csv")

# Drop NaNs / empty
df = df.dropna(subset=["clean_headline", "category"])
df = df[df["clean_headline"].str.strip() != ""]

# ----- LABEL MAPPING (5 → 3 classes) -----
def map_label(label):
    if label in ["Left", "Left-Center"]:
        return "Left"
    elif label == "Center":
        return "Neutral"
    else:  # Center-Right, Right
        return "Right"

df["bias"] = df["category"].apply(map_label)

print("Class distribution:")
print(df["bias"].value_counts())

# Features / target
X = df["clean_headline"].astype(str)
y = df["bias"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF (add trigrams + balance)
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

print("\n==== 3-CLASS CLASSIFICATION REPORT ====")
print(classification_report(y_test, y_pred))

print("\n==== CONFUSION MATRIX ====")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/bias_model_3class.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer_3class.pkl")

print("\nSaved:")
print("models/bias_model_3class.pkl")
print("models/tfidf_vectorizer_3class.pkl")
