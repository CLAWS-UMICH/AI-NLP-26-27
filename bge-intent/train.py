"""Fine-tune BAAI/bge-small-en-v1.5 as a single-label intent classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MODEL_NAME, IntentClassifier, get_device, load_tokenizer, tokenize


SEED = 42


def load_examples(path: Path):
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        rows = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    texts, labels, groups = [], [], []
    for number, row in enumerate(rows, 1):
        text = row.get("text")
        label = row.get("intent")
        if label is None and isinstance(row.get("intents"), list) and len(row["intents"]) == 1:
            label = row["intents"][0]
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
            raise ValueError(f"row {number}: expected non-empty text and one intent")
        texts.append(text.strip())
        labels.append(label)
        group = row.get("family")
        groups.append(group if isinstance(group, str) and group else f"row_{number}")
    if not texts:
        raise ValueError(f"{path} contains no examples")
    return texts, labels, groups


def grouped_split(groups: list[str]):
    """Keep paraphrase families together; training_only families never enter evaluation."""
    train_only = {g for g in groups if g.startswith("training_only_")}
    available = sorted(set(groups) - train_only)
    if len(available) < 5:
        raise ValueError("need at least five phrase families for grouped train/validation/test splits")
    shuffled = list(np.random.default_rng(SEED).permutation(available))
    test_groups = set(shuffled[: max(1, round(len(shuffled) * 0.15))])
    validation_groups = set(shuffled[max(1, round(len(shuffled) * 0.15)):][: max(1, round(len(shuffled) * 0.15))])
    train_groups = set(shuffled) - test_groups - validation_groups | train_only
    return tuple(np.asarray([i for i, group in enumerate(groups) if group in selected])
                 for selected in (train_groups, validation_groups, test_groups))


class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def make_collate(tokenizer, device):
    def collate(rows):
        texts, labels = zip(*rows)
        return tokenize(tokenizer, list(texts), device), torch.tensor(labels, dtype=torch.long, device=device)
    return collate


def predict(model, tokenizer, texts, device, batch_size):
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenize(tokenizer, texts[start:start + batch_size], device)
            results.append(model(**inputs).softmax(dim=1).cpu().numpy())
    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("trainingData.json"))
    parser.add_argument("--validation-data", type=Path,
                        help="use an explicit validation JSON/JSONL file")
    parser.add_argument("--test-data", type=Path,
                        help="use an explicit test JSON/JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size, patience, and learning rate must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if bool(args.validation_data) != bool(args.test_data):
        parser.error("--validation-data and --test-data must be provided together")

    texts, labels, groups = load_examples(args.data)
    label_names = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(label_names)}
    ids = np.asarray([label2id[label] for label in labels])
    if args.validation_data and args.test_data:
        val_texts, val_labels, _ = load_examples(args.validation_data)
        test_texts, test_labels, _ = load_examples(args.test_data)
        unknown = (set(val_labels) | set(test_labels)) - set(label2id)
        if unknown:
            raise ValueError(f"validation/test contain labels absent from training: {sorted(unknown)}")
        train_texts, train_ids = texts, ids
        val_ids = np.asarray([label2id[label] for label in val_labels])
        test_ids = np.asarray([label2id[label] for label in test_labels])
        train_families = validation_families = test_families = []
        split_strategy = "explicit_files"
    else:
        train_idx, val_idx, test_idx = grouped_split(groups)
        take = lambda idx: ([texts[i] for i in idx], ids[idx])
        train_texts, train_ids = take(train_idx)
        val_texts, val_ids = take(val_idx)
        test_texts, test_ids = take(test_idx)
        train_families = sorted({groups[i] for i in train_idx})
        validation_families = sorted({groups[i] for i in val_idx})
        test_families = sorted({groups[i] for i in test_idx})
        split_strategy = "grouped_by_phrase_family"

    device = get_device()
    print(f"Loading {MODEL_NAME} on {device}")
    tokenizer = load_tokenizer()
    model = IntentClassifier(len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(IntentDataset(train_texts, train_ids.tolist()), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(SEED),
                        collate_fn=make_collate(tokenizer, device))
    best_state, best_score, best_epoch, stale = None, (-1.0, -1.0), 0, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        val_logits = []
        with torch.inference_mode():
            for start in range(0, len(val_texts), args.batch_size):
                batch_texts = val_texts[start:start + args.batch_size]
                batch_ids = torch.as_tensor(val_ids[start:start + args.batch_size], dtype=torch.long, device=device)
                logits = model(**tokenize(tokenizer, batch_texts, device))
                val_losses.append(criterion(logits, batch_ids).item())
                val_logits.append(logits.softmax(dim=1).cpu().numpy())
        val_probs = np.concatenate(val_logits)
        val_preds = val_probs.argmax(axis=1)
        score = (accuracy_score(val_ids, val_preds), f1_score(val_ids, val_preds, average="macro", zero_division=0))
        epoch_row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                     "validation_loss": float(np.mean(val_losses)),
                     "validation_accuracy": float(score[0]), "validation_macro_f1": float(score[1])}
        history.append(epoch_row)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={epoch_row['train_loss']:.4f}, validation loss={epoch_row['validation_loss']:.4f}, validation accuracy={score[0]:.4f}, macro-F1={score[1]:.4f}")
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    test_probs = predict(model, tokenizer, test_texts, device, args.batch_size)
    test_preds = test_probs.argmax(axis=1)
    accuracy = accuracy_score(test_ids, test_preds)
    macro_f1 = f1_score(test_ids, test_preds, average="macro", zero_division=0)
    report = classification_report(test_ids, test_preds, labels=list(range(len(label_names))),
                                   target_names=label_names, zero_division=0)
    print(f"Held-out test: accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}\n{report}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "label2id.json").write_text(json.dumps(label2id, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "model": MODEL_NAME, "accuracy": accuracy, "macro_f1": macro_f1,
        "validation_accuracy": best_score[0], "validation_macro_f1": best_score[1],
        "best_epoch": best_epoch, "epochs_requested": args.epochs,
        "train_rows": len(train_texts), "validation_rows": len(val_texts), "test_rows": len(test_texts),
        "split_strategy": split_strategy,
        "history": history,
        "train_families": train_families,
        "validation_families": validation_families,
        "test_families": test_families,
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved BGE intent model and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
