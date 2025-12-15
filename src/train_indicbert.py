import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "bert-base-multilingual-cased"

NUM_LABELS = 3

# Load data
df = pd.read_csv("data/processed/india_clean_dataset.csv")

df = df.dropna(subset=["clean_headline", "category"])
df = df[df["clean_headline"].str.strip() != ""]

def map_label(label):
    if label in ["Left", "Left-Center"]:
        return 0   # Left
    elif label == "Center":
        return 1   # Neutral
    else:
        return 2   # Right

df["label"] = df["category"].apply(map_label)

dataset = Dataset.from_pandas(df[["clean_headline", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["clean_headline"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

args = TrainingArguments(
    output_dir="models/indicbert_bias",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
