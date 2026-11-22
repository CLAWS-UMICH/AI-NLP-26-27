"""
Stage 2: fine-tune the (primed) encoder + head to classify intents.

1. Encoder: newest runs/prime-*/encoder if there is one, else base MiniLM (override with --encoder)
2. Train on data/train.jsonl; val split is grouped (see data.split) so augmented
   copies of one seed row never sit on both sides. Every sentence goes through glossary normalization first
3. Cross-entropy with label smoothing, AdamW with a separate (bigger) LR for the random head,
   linear warmup + decay, grad clipping
4. After each epoch: val loss / accuracy / macro-F1. Keeps the weights from the lowest val-loss epoch
   and stops early after `patience` epochs without improvement
5. Scores data/test.jsonl ONCE with the kept weights. Only the miss count is printed so the test set
   doesn't end up steering data fixes (val misses are printed and saved instead)
6. Writes everything to runs/train-<time>[-tag]/: model.pt, label2id.json, encoder_config/, tokenizer/,
   glossary.json, config.json, history.json, metrics.json, val_misses.txt, learning_curves.png

Usage:
    python code/train.py
    python code/train.py --encoder base --tag no-prime
    python code/train.py --encoder runs/prime-20261114-120000/encoder --seed 1
"""

import argparse
import copy
import random
import shutil
from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from config import BASE_MODEL, GLOSSARY_FILE, MIN_CONF, RUNS, SEED, TRAIN, UNHANDLED
from data import load_labels, load_test, load_train, split
from normalize import Normalizer
from model import IntentModel, load_tokenizer, pick_device, tokenize
from plot_curves import plot
from rundir import new_run, write_json

EVAL_BATCH = 128


def resolve_encoder(choice):
    if choice == "base":
        return BASE_MODEL
    if choice != "auto":
        return choice
    primed = sorted(p / "encoder" for p in RUNS.glob("prime-*") if (p / "encoder" / "config.json").exists())
    return str(primed[-1]) if primed else BASE_MODEL


@torch.inference_mode()
def evaluate(model, tokenizer, texts, ids, device, unhandled_id):
    """Predicted ids (low confidence -> unhandled) and mean unsmoothed cross-entropy."""
    model.eval()
    preds, loss = [], 0.0
    for i in range(0, len(texts), EVAL_BATCH):
        logits = model(tokenize(tokenizer, texts[i:i + EVAL_BATCH], device))
        target = torch.tensor(ids[i:i + EVAL_BATCH], device=device)
        loss += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
        conf, idx = logits.softmax(dim=1).max(dim=1)
        idx[conf < MIN_CONF] = unhandled_id
        preds.extend(idx.tolist())
    return preds, loss / len(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="auto", help="auto (newest primed, else base) | base | <path or HF name>")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    encoder_name = resolve_encoder(args.encoder)

    rows = load_train()
    train_rows, val_rows = split(rows)
    test_rows = load_test(rows)

    labels = load_labels()
    label2id = {name: i for i, name in enumerate(labels)}
    unhandled_id = label2id[UNHANDLED]
    unknown = {r["intent"] for r in rows + test_rows} - set(labels)
    if unknown:
        raise SystemExit(f"labels not in intents.json: {sorted(unknown)} (run python code/validate.py)")

    normalize = Normalizer()  # same rewrite at train and predict time (predict loads the run's copy)

    def unpack(rs):
        return [normalize(r["text"]) for r in rs], [label2id[r["intent"]] for r in rs]

    train_x, train_ids = unpack(train_rows)
    val_x, val_ids = unpack(val_rows)
    test_x, test_ids = unpack(test_rows)
    print(f"{encoder_name} on {device}: {len(train_x)} train / {len(val_x)} val / {len(test_x)} test, {len(labels)} intents")

    tokenizer = load_tokenizer(encoder_name)
    model = IntentModel.from_pretrained(encoder_name, len(labels), TRAIN["dropout"]).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": TRAIN["encoder_lr"]},
        {"params": model.head.parameters(), "lr": TRAIN["head_lr"]},
    ], weight_decay=TRAIN["weight_decay"])
    bs = TRAIN["batch"]
    steps = TRAIN["epochs"] * ((len(train_x) + bs - 1) // bs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(TRAIN["warmup"] * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=TRAIN["label_smoothing"])

    history, best, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(1, TRAIN["epochs"] + 1):
        model.train()
        order = list(range(len(train_x)))
        random.shuffle(order)
        total = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            batch = tokenize(tokenizer, [train_x[j] for j in idx], device)
            target = torch.tensor([train_ids[j] for j in idx], device=device)
            loss = loss_fn(model(batch), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)

        val_pred, val_loss = evaluate(model, tokenizer, val_x, val_ids, device, unhandled_id)
        entry = {
            "epoch": epoch,
            "train_loss": total / len(train_x),
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(val_ids, val_pred),
            "val_macro_f1": f1_score(val_ids, val_pred, average="macro"),
        }
        history.append(entry)
        print(f"epoch {epoch:2d}  loss {entry['train_loss']:.4f}  val-loss {val_loss:.4f}  "
              f"val-acc {entry['val_accuracy']:.4f}  val-macro-F1 {entry['val_macro_f1']:.4f}")

        if val_loss < best_loss:
            best_loss, best, stale = val_loss, (epoch, copy.deepcopy(model.state_dict())), 0
        else:
            stale += 1
            if stale >= TRAIN["patience"]:
                print(f"early stop: no val-loss improvement for {stale} epochs")
                break

    best_epoch, state = best
    model.load_state_dict(state)
    print(f"\nkeeping epoch {best_epoch} (val-loss {best_loss:.4f})")

    val_pred, _ = evaluate(model, tokenizer, val_x, val_ids, device, unhandled_id)
    val_misses = [f"{t!r}  expected {labels[g]}  got {labels[p]}" for t, g, p in zip(val_x, val_ids, val_pred) if g != p]
    print(f"{len(val_misses)} validation misses out of {len(val_x)}")
    for line in val_misses:
        print("  MISS", line)

    pred, _ = evaluate(model, tokenizer, test_x, test_ids, device, unhandled_id)
    by_style = defaultdict(list)
    for r, g, p in zip(test_rows, test_ids, pred):
        by_style[r.get("style", "all")].append(g == p)
    test_misses = sum(g != p for g, p in zip(test_ids, pred))
    metrics = {
        "encoder": encoder_name,
        "best_epoch": best_epoch,
        "accuracy": accuracy_score(test_ids, pred),
        "macro_f1": f1_score(test_ids, pred, average="macro"),
        "test_misses": test_misses,
        "accuracy_by_style": {k: sum(v) / len(v) for k, v in sorted(by_style.items())},
        "val_accuracy": accuracy_score(val_ids, val_pred),
        "val_macro_f1": f1_score(val_ids, val_pred, average="macro"),
        "train_rows": len(train_x),
        "val_rows": len(val_x),
        "test_rows": len(test_x),
    }
    print(f"\nTEST ({len(test_x)})  acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}  "
          f"({test_misses} misses)  by style: {metrics['accuracy_by_style']}")

    run = new_run("train", args.tag)
    torch.save(model.state_dict(), run / "model.pt")
    model.encoder.config.save_pretrained(run / "encoder_config")
    tokenizer.save_pretrained(run / "tokenizer")
    if normalize:
        shutil.copy(GLOSSARY_FILE, run / "glossary.json")
    write_json(run / "label2id.json", label2id)
    write_json(run / "config.json", {"encoder": encoder_name, "seed": args.seed, "min_conf": MIN_CONF, "normalize": bool(normalize), **TRAIN})
    write_json(run / "history.json", history)
    write_json(run / "metrics.json", metrics)
    (run / "val_misses.txt").write_text("\n".join(val_misses) + "\n", encoding="utf-8")
    plot(run)
    print(f"saved {run}")


if __name__ == "__main__":
    main()
