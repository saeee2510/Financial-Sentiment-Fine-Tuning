# Financial Sentiment: Results

Train 4,000 · validation 955 · test 2,388 headlines

| Model | Accuracy | Macro-F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.772 | 0.689 | 0.562 | 0.645 | 0.861 |
| Fine-tuned distilbert-base-uncased | 0.849 | 0.797 | 0.721 | 0.768 | 0.901 |

Fine-tuning changed macro-F1 by **+0.107** over the baseline.

## Most common confusions (last model)

- Bullish predicted as Neutral: 97
- Neutral predicted as Bearish: 76
- Bearish predicted as Neutral: 70

## Training

Device: mps · time: 351.6s

- Epoch 1: train loss 0.6744, validation macro-F1 0.7394
- Epoch 2: train loss 0.3558, validation macro-F1 0.7915

High-confidence mistakes are in `output/errors.csv`.
