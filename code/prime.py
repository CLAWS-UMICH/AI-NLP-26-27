"""
Stage 1: prime the encoder on astronaut language before it ever sees an intent label.

- Reads data/astronaut_primer.json: a list of {"anchor": jargon/callout, "positive": plain-English meaning}
- Contrastive training (multiple-negatives ranking loss, both directions): each anchor must pick its own
  positive out of the whole batch, so "bird's drifting east" ends up near "the satellite is drifting east"
- Holds out 10% of pairs and reports anchor->positive retrieval accuracy before and after,
  so we can see whether priming actually taught the encoder anything
- Saves the primed encoder + tokenizer to runs/prime-<time>/encoder/. train.py picks up the newest one.

Usage:
    python code/prime.py
    python code/prime.py --tag nato-only
    python code/prime.py --epochs 10
"""

import argparse
import random

import torch
import torch.nn.functional as F
from transformers import AutoModel, get_linear_schedule_with_warmup

from config import BASE_MODEL, PRIME, SEED
from data import load_primer
from model import load_tokenizer, mean_pool, pick_device, tokenize
from rundir import new_run, write_json

HOLDOUT = 0.1


def embed(encoder, tokenizer, texts, device):
    batch = tokenize(tokenizer, texts, device)
    return F.normalize(mean_pool(encoder(**batch).last_hidden_state, batch["attention_mask"]), dim=-1)


@torch.inference_mode()
def retrieval_acc(encoder, tokenizer, pairs, device):
    """Share of held-out anchors whose nearest positive (among all held-out positives) is their own."""
    encoder.eval()
    a = torch.cat([embed(encoder, tokenizer, [x for x, _ in pairs[i:i + 128]], device) for i in range(0, len(pairs), 128)])
    p = torch.cat([embed(encoder, tokenizer, [y for _, y in pairs[i:i + 128]], device) for i in range(0, len(pairs), 128)])
    hits = (a @ p.T).argmax(dim=1) == torch.arange(len(pairs), device=device)
    return hits.float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument("--epochs", type=int, default=PRIME["epochs"])
    args = parser.parse_args()
    PRIME["epochs"] = args.epochs

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()

    pairs = load_primer()
    random.shuffle(pairs)
    n_hold = max(1, int(len(pairs) * HOLDOUT))
    held, pairs = pairs[:n_hold], pairs[n_hold:]
    print(f"priming {BASE_MODEL} on {device}: {len(pairs)} pairs ({len(held)} held out)")

    tokenizer = load_tokenizer(BASE_MODEL)
    encoder = AutoModel.from_pretrained(BASE_MODEL).to(device)
    before = retrieval_acc(encoder, tokenizer, held, device)
    print(f"held-out retrieval before: {before:.3f}")

    bs = PRIME["batch"]
    steps = PRIME["epochs"] * ((len(pairs) + bs - 1) // bs)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=PRIME["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(PRIME["warmup"] * steps), steps)

    history = []
    for epoch in range(1, PRIME["epochs"] + 1):
        encoder.train()
        random.shuffle(pairs)
        total = 0.0
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            if len(chunk) < 2:  # a lone pair has no negatives to contrast against
                continue
            a = embed(encoder, tokenizer, [x for x, _ in chunk], device)
            p = embed(encoder, tokenizer, [y for _, y in chunk], device)
            sims = a @ p.T * PRIME["scale"]
            target = torch.arange(len(chunk), device=device)
            loss = (F.cross_entropy(sims, target) + F.cross_entropy(sims.T, target)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(chunk)
        acc = retrieval_acc(encoder, tokenizer, held, device)
        history.append({"epoch": epoch, "loss": total / len(pairs), "heldout_retrieval": acc})
        print(f"epoch {epoch}  loss {total / len(pairs):.4f}  held-out retrieval {acc:.3f}")

    run = new_run("prime", args.tag)
    encoder.save_pretrained(run / "encoder")
    tokenizer.save_pretrained(run / "encoder")
    write_json(run / "history.json", history)
    write_json(run / "metrics.json", {
        "base_model": BASE_MODEL,
        "pairs": len(pairs),
        "heldout_pairs": len(held),
        "heldout_retrieval_before": before,
        "heldout_retrieval_after": history[-1]["heldout_retrieval"],
        "seed": SEED,
        **PRIME,
    })
    print(f"saved {run}")


if __name__ == "__main__":
    main()
