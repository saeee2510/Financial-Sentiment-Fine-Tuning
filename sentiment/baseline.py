"""Baseline: TF-IDF features (words + bigrams) with class-balanced logistic regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def train_baseline(train_df, seed=42):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(train_df["text"], train_df["label"])
    return model


def predict(model, texts):
    probs = model.predict_proba(texts)
    return probs.argmax(axis=1), probs.max(axis=1)
