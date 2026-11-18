"""
Trains the TF-IDF baseline and the fine-tuned transformer, evaluates both on the
held-out test set, and writes a report with metrics, confusion matrices and errors.

Run:  python -m sentiment.run                  # default: 4,000 training headlines, 2 epochs
      python -m sentiment.run --train-size 0   # use the full training split
      python -m sentiment.run --skip-finetune  # baseline only (no torch needed)
"""
import argparse
import json
import time

from . import baseline, evaluate
from .data import LABELS, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--skip-finetune", action="store_true")
    args = ap.parse_args()

    train_df, val_df, test_df = load(args.csv, train_size=args.train_size or None)
    print(f"train {len(train_df)}, validation {len(val_df)}, test {len(test_df)}")
    results, extras = {}, {}

    t0 = time.time()
    base = baseline.train_baseline(train_df)
    b_pred, b_conf = baseline.predict(base, test_df["text"])
    results["TF-IDF + logistic regression"] = evaluate.metrics(test_df["label"], b_pred)
    extras["baseline_seconds"] = round(time.time() - t0, 1)

    if not args.skip_finetune:
        from . import finetune
        t0 = time.time()
        model, tok, device, history = finetune.finetune(train_df, val_df, args.model, epochs=args.epochs)
        f_pred, f_conf = finetune.predict(model, tok, device, test_df["text"])
        results[f"Fine-tuned {args.model}"] = evaluate.metrics(test_df["label"], f_pred)
        extras.update({"finetune_seconds": round(time.time() - t0, 1), "device": str(device), "history": history})
        evaluate.errors_table(test_df["text"], test_df["label"], f_pred, f_conf).to_csv("output/errors.csv", index=False)
    else:
        evaluate.errors_table(test_df["text"], test_df["label"], b_pred, b_conf).to_csv("output/errors.csv", index=False)

    out = {"sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
           "results": results, **extras}
    with open("output/results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    evaluate.plot_confusions(results, "output/confusion_matrices.png")
    write_report(out)


def write_report(out):
    lines = ["# Financial Sentiment: Results", "",
             f"Train {out['sizes']['train']:,} · validation {out['sizes']['validation']:,} · "
             f"test {out['sizes']['test']:,} headlines", "",
             "| Model | Accuracy | Macro-F1 | " + " | ".join(f"{l} F1" for l in LABELS) + " |",
             "|---|---|---|" + "---|" * len(LABELS)]
    for name, m in out["results"].items():
        per = " | ".join(f"{m['per_class'][l]['f1']:.3f}" for l in LABELS)
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {per} |")
    names = list(out["results"])
    if len(names) == 2:
        gain = out["results"][names[1]]["macro_f1"] - out["results"][names[0]]["macro_f1"]
        lines += ["", f"Fine-tuning changed macro-F1 by **{gain:+.3f}** over the baseline."]
    lines += ["", "## Most common confusions (last model)", ""]
    for c in evaluate.top_confusions(out["results"][names[-1]]["confusion_matrix"]):
        lines.append(f"- {c['true']} predicted as {c['predicted']}: {c['count']}")
    if "history" in out:
        lines += ["", "## Training", "", f"Device: {out['device']} · time: {out['finetune_seconds']}s", ""]
        lines += [f"- Epoch {h['epoch']}: train loss {h['train_loss']:.4f}, validation macro-F1 {h['val_macro_f1']:.4f}"
                  for h in out["history"]]
    lines += ["", "High-confidence mistakes are in `output/errors.csv`."]
    with open("output/report.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
