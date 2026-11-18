# Financial Sentiment: Fine-Tuning a Transformer vs. a Classic Baseline

Classifies financial news headlines as **Bearish**, **Bullish** or **Neutral**. I fine-tuned DistilBERT and compared it against a TF-IDF + logistic regression baseline, to see how much the transformer actually adds and where both still get it wrong.

Data: the public [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset (~9.5K training and ~2.4K test headlines). Its validation split is the held-out test set, and 10% of the training split is used to pick the best epoch.

## Results

<!-- Paste the table from output/report.md after running -->

| Model | Accuracy | Macro-F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | | | | | |
| Fine-tuned DistilBERT | | | | | |

![Confusion matrices](output/confusion_matrices.png)

**What I noticed:** <!-- after running, write 2-3 sentences: which classes improved most, which confusions remain, and one example from output/errors.csv -->

## How it works

| File | What it does |
|---|---|
| `sentiment/data.py` | Loads the dataset (or your own CSV) and makes stratified train / validation / test splits |
| `sentiment/baseline.py` | TF-IDF (words + bigrams) with class-balanced logistic regression |
| `sentiment/finetune.py` | Fine-tunes DistilBERT with a plain PyTorch loop: AdamW, warmup + linear decay, gradient clipping, keeps the epoch with the best validation macro-F1 |
| `sentiment/evaluate.py` | Accuracy, macro-F1, per-class precision/recall, confusion matrices, most confident mistakes |
| `sentiment/run.py` | Runs both models and writes `output/report.md`, `results.json`, `confusion_matrices.png`, `errors.csv` |

Macro-F1 is the main metric because the classes are imbalanced (most headlines are Neutral), so accuracy alone would flatter a model that leans Neutral.

## Run

```bash
pip install -r requirements.txt
python -m sentiment.run                  # 4,000 training headlines, 2 epochs (~20-40 min on a laptop CPU, faster on Apple Silicon)
python -m sentiment.run --train-size 0   # full training set
python -m sentiment.run --skip-finetune  # baseline only
python -m unittest discover -s tests -t .
```

Runs on Apple Silicon (MPS), a GPU, or CPU automatically.

## Limitations

- Headlines are short and often ambiguous; some "errors" are labels a human would argue with.
- Trained on a subset by default to keep runtime reasonable on a laptop.
- No hyperparameter search beyond picking the best epoch.
# Financial-Sentiment-Fine-Tuning
