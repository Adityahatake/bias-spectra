import joblib
from political_filter import is_political

model = joblib.load("models/bias_model_3class.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_3class.pkl")

POLITICAL_KEYWORDS = [
    "government", "govt", "bjp", "congress", "pm", "prime minister",
    "modi", "rahul", "election", "poll", "vote", "parliament",
    "supreme court", "sc", "high court", "policy", "law",
    "minister", "cabinet", "budget", "tax", "reservation",
    "protest", "farmer", "caa", "nrc", "article 370" 
]

def is_political(text):
    text = text.lower()
    for kw in POLITICAL_KEYWORDS:
        if kw in text:
            return True
    return False


while True:
    text = input("\nEnter headline (or exit): ")
    if text.lower() == "exit":
        break

    if not is_political(text):
        print("Predicted Bias: Neutral (Non-Political)")
        continue

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    print("Predicted Bias:", pred)
