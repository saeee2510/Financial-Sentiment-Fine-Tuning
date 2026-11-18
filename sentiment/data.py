"""
Loads financial news headlines labeled Bearish / Bullish / Neutral.

Default source: the public Hugging Face dataset "zeroshot/twitter-financial-news-sentiment"
(~9.5K training and ~2.4K validation headlines). Its validation split is used as the
held-out test set; a stratified slice of the training split is used for model selection.
You can also point --csv at your own file with "text" and "label" columns.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Bearish", "Bullish", "Neutral"]
HF_DATASET = "zeroshot/twitter-financial-news-sentiment"


def load(csv_path=None, train_size=None, seed=42):
    if csv_path:
        df = pd.read_csv(csv_path)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    else:
        from datasets import load_dataset  # imported here so tests run without it
        ds = load_dataset(HF_DATASET)
        train_df, test_df = ds["train"].to_pandas(), ds["validation"].to_pandas()

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=seed)
    if train_size and train_size < len(train_df):
        train_df, _ = train_test_split(train_df, train_size=train_size, stratify=train_df["label"], random_state=seed)
    clean = lambda d: d[["text", "label"]].reset_index(drop=True)
    return clean(train_df), clean(val_df), clean(test_df)
