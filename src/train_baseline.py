import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("data/processed/india_clean_dataset.csv")
# Remove empty / NaN headlines
df = df.dropna(subset=["clean_headline", "category"])
df = df[df["clean_headline"].str.strip() != ""]

X = df["clean_headline"
].astype(str)
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF + Logistic Regression Pipeline
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),    # bigrams improve political bias detection
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=3000)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

print("\n==== CLASSIFICATION REPORT ====")
print(classification_report(y_test, y_pred))

print("\n==== CONFUSION MATRIX ====")
print(confusion_matrix(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "models/bias_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved to models/bias_model.pkl")
print("Vectorizer saved to models/tfidf_vectorizer.pkl")
