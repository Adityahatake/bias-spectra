# src/preprocess_india_full.py
import pandas as pd
import re
import math
from collections import Counter

RAW_CSV = "data/raw/india_news_raw.csv"
OUT_CSV = "data/processed/india_clean_dataset.csv"
TARGET_TOTAL = 1500  # final target; script will try to balance

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.lower()
    # remove URLs and unusual characters but keep common punctuation minimal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv(RAW_CSV)
    print("Raw rows:", len(df))
    # drop empty headlines 
    df = df[df['headline'].notna()]
    df['clean_headline'] = df['headline'].apply(clean_text)

    # normalize category names (consistent labels)
    df['category'] = df['category'].str.strip()

    # Show counts
    counts = df['category'].value_counts().to_dict()
    print("Counts before balancing:", counts)

    # Decide per-category target (equalize across categories)
    categories = sorted(df['category'].unique())
    n_cat = len(categories)
    per_cat = math.ceil(TARGET_TOTAL / n_cat)
    print(f"Balancing: {TARGET_TOTAL} total → {per_cat} per category (categories: {categories})")

    samples = []
    for cat in categories:
        sub = df[df['category'] == cat]
        if len(sub) >= per_cat:
            sampled = sub.sample(n=per_cat, random_state=42)
        else:
            # if not enough, take all and later we'll fill from other categories
            sampled = sub.copy()
        samples.append(sampled)

    balanced = pd.concat(samples, ignore_index=True)
    # if still short of TARGET_TOTAL, fill from the largest categories
    if len(balanced) < TARGET_TOTAL:
        needed = TARGET_TOTAL - len(balanced)
        print("Filling shortage of", needed, "from larger categories.")
        remaining = df[~df.index.isin(balanced.index)]
        if len(remaining) > 0:
            fill = remaining.sample(n=min(needed, len(remaining)), random_state=42)
            balanced = pd.concat([balanced, fill], ignore_index=True)

    # Final dedupe (headline)
    balanced = balanced.drop_duplicates(subset=['clean_headline'])
    print("Final dataset rows after balancing & dedupe:", len(balanced))

    # Create simplified label columns (optional)
    # Keep original category taxonomy:
    # Left, Left-Center, Center, Center-Right, Right
    balanced = balanced[['headline', 'clean_headline', 'url', 'source', 'category']]

    balanced.to_csv(OUT_CSV, index=False)
    print("Saved processed dataset to:", OUT_CSV)
    print("Category distribution now:\n", balanced['category'].value_counts().to_dict())

if __name__ == "__main__":
    main()
