"""Continue pretraining BGE on astronaut-domain anchor/positive pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from model import MODEL_NAME, get_device


SEED = 42


class PrimerPairs(Dataset):
    def __init__(self, path: Path):
        self.rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError("primer must be a non-empty JSON list")
        for i, row in enumerate(self.rows, 1):
            if not all(isinstance(row.get(key), str) and row[key].strip()
                       for key in ("anchor", "positive")):
                raise ValueError(f"row {i}: expected non-empty anchor and positive strings")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row["anchor"].strip(), row["positive"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("../astronaut_primer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/astronaut-primer-encoder"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("epochs, batch size, learning rate, and temperature must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Loading {MODEL_NAME} on {device}")
    # Keep training reproducible and runnable in restricted/offline environments.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    dataset = PrimerPairs(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), drop_last=False)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = []

    for epoch in range(args.epochs):
        encoder.train()
        losses = []
        for anchors, positives in loader:
            texts = list(anchors) + list(positives)
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = encoder(**inputs).last_hidden_state[:, 0]
            vectors = F.normalize(outputs, p=2, dim=1)
            count = len(anchors)
            logits = vectors[:count] @ vectors[count:].T / args.temperature
            targets = torch.arange(count, device=device)
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        print(f"epoch {epoch + 1}/{args.epochs}: contrastive loss={mean_loss:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder.cpu().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    report = {
        "base_model": MODEL_NAME,
        "objective": "symmetric in-batch anchor-positive contrastive loss",
        "training_file": str(args.data),
        "pairs": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "epoch_mean_losses": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved astronaut-domain encoder and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
