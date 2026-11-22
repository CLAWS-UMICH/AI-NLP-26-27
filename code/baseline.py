"""
Step 0: baseline, no fine-tuning. The number train.py has to beat.

- Same train/val split as train.py (data.split), fits on the train part only
- Same glossary normalization as train.py, then frozen encoder embeddings + logistic regression,
  scored on data/test.jsonl
- Writes runs/baseline-<time>[-tag]/: report.txt, metrics.json

Usage:
    python code/baseline.py
    python code/baseline.py --encoder runs/prime-.../encoder --tag primed    (did priming help the raw embeddings?)
"""

import argparse

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import BASE_MODEL, SEED
from data import load_test, load_train, split
from normalize import Normalizer
from rundir import new_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=BASE_MODEL)
    parser.add_argument("--tag")
    args = parser.parse_args()

    rows = load_train()
    train_rows, _ = split(rows)
    test_rows = load_test(rows)
    normalize = Normalizer()
    train_x, train_y = [normalize(r["text"]) for r in train_rows], [r["intent"] for r in train_rows]
    test_x, test_y = [normalize(r["text"]) for r in test_rows], [r["intent"] for r in test_rows]
    print(f"{args.encoder}: {len(train_x)} train / {len(test_x)} test")

    encoder = SentenceTransformer(args.encoder)
    train_vecs = encoder.encode(train_x, batch_size=128, normalize_embeddings=True)
    test_vecs = encoder.encode(test_x, batch_size=128, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=3000, C=10, random_state=SEED)  # normalized vectors are small, so weaker regularization
    clf.fit(train_vecs, train_y)
    pred = clf.predict(test_vecs)

    acc, macro_f1 = accuracy_score(test_y, pred), f1_score(test_y, pred, average="macro")
    report = (
        f"encoder: {args.encoder}\n"
        f"accuracy: {acc:.4f}  macro-F1: {macro_f1:.4f}  (train {len(train_x)}, test {len(test_x)})\n\n"
        + classification_report(test_y, pred, zero_division=0)
    )
    print(report)

    run = new_run("baseline", args.tag)
    (run / "report.txt").write_text(report, encoding="utf-8")
    write_json(run / "metrics.json", {"encoder": args.encoder, "accuracy": acc, "macro_f1": macro_f1,
                                      "train_rows": len(train_x), "test_rows": len(test_x)})
    print(f"saved {run}")


if __name__ == "__main__":
    main()
