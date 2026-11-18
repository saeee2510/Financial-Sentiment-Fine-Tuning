"""Tests that run without torch, transformers or internet access."""
import os
import tempfile
import unittest

import pandas as pd

from sentiment import baseline, evaluate
from sentiment.data import LABELS, load

BEAR = ["shares plunge after weak earnings", "stock falls on downgrade", "revenue misses estimates, shares drop",
        "company cuts guidance, stock slides", "losses widen as sales decline"]
BULL = ["shares surge on record profit", "stock jumps after upgrade", "earnings beat estimates, shares rise",
        "company raises guidance, stock climbs", "revenue soars on strong demand"]
NEUT = ["company to report earnings next week", "board schedules annual meeting", "ceo to speak at conference",
        "firm files quarterly report", "company announces new office location"]


def toy_frame(copies=8):
    rows = []
    for _ in range(copies):
        rows += [(t, 0) for t in BEAR] + [(t, 1) for t in BULL] + [(t, 2) for t in NEUT]
    return pd.DataFrame(rows, columns=["text", "label"])


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        toy_frame().to_csv(self.tmp.name, index=False)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_load_splits_are_disjoint_and_stratified(self):
        train, val, test = load(self.tmp.name)
        self.assertEqual(len(train) + len(val) + len(test), len(toy_frame()))
        self.assertEqual(set(test["label"]), {0, 1, 2})

    def test_baseline_learns_clear_signal(self):
        train, _, test = load(self.tmp.name)
        model = baseline.train_baseline(train)
        pred, conf = baseline.predict(model, test["text"])
        self.assertGreater(evaluate.metrics(test["label"], pred)["macro_f1"], 0.9)
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())

    def test_metrics_and_confusions(self):
        m = evaluate.metrics([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertEqual(m["confusion_matrix"][0][1], 1)
        top = evaluate.top_confusions(m["confusion_matrix"], k=1)[0]
        self.assertEqual((top["true"], top["predicted"]), ("Bearish", "Bullish"))

    def test_errors_table_only_has_mistakes(self):
        errs = evaluate.errors_table(["a", "b", "c"], [0, 1, 2], [0, 2, 2], [0.9, 0.8, 0.7])
        self.assertEqual(list(errs["text"]), ["b"])
        self.assertEqual(LABELS[2], errs.iloc[0]["predicted"])


if __name__ == "__main__":
    unittest.main()
