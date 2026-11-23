"""Train MiniLM (intent head + B/I/O slot head) and evaluate on dev (DATA_GENERATION.md §6).
Usage: python -m datagen.train_eval --round K [--primer] [--epochs 6]"""
import argparse, json, math, random, re, time
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from .config import GEN, DEV_HAND, DEV_HARD, DEV_HARD_ASR, REPORTS, RUNS, LABELS, FAMILY_OF, FAMILIES, SLOT_LABELS, STUDENT, DATA
from .common import read_jsonl, write_jsonl, norm
from . import diversity

L2I = {l: i for i, l in enumerate(LABELS)}
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def slot_span(text_n, slot):
    if not slot: return None
    m = re.search(rf"\b{re.escape(norm(slot))}\b", text_n)
    return (m.start(), m.end()) if m else None

class Net(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc; h = enc.config.hidden_size
        self.drop = nn.Dropout(0.1); self.intent = nn.Linear(h, len(LABELS)); self.slot = nn.Linear(h, 3)
    def forward(self, ids, mask):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (hs * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.intent(self.drop(pooled)), self.slot(self.drop(hs))

def batches(tok, rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [rows[j] for j in idx[i:i + bs]]
        texts = [norm(r["text"]) for r in b]
        enc = tok(texts, padding=True, truncation=True, max_length=32, return_offsets_mapping=True, return_tensors="pt")
        bio = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for k, r in enumerate(b):
            span = slot_span(texts[k], r.get("slot_text")) if r["label"] in SLOT_LABELS else None
            first = True
            for t, (s, e) in enumerate(enc["offset_mapping"][k].tolist()):
                if s == e == 0: continue
                if span and s >= span[0] and e <= span[1]:
                    bio[k, t] = 1 if first else 2; first = False
                else:
                    bio[k, t] = 0
        y = torch.tensor([L2I[r["label"]] for r in b])
        yield b, texts, enc, bio, y

def primer_encoder(tok_name, rev):
    """Domain adaptation on data/astronaut_primer.json (anchor/positive pairs) with MultipleNegativesRankingLoss."""
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from datasets import Dataset
    p = json.load(open(DATA / "astronaut_primer.json"))
    ds = Dataset.from_dict({"anchor": [norm(x["anchor"]) for x in p], "positive": [norm(x["positive"]) for x in p]})
    st = SentenceTransformer(tok_name, revision=rev, device=DEV)
    args = SentenceTransformerTrainingArguments(output_dir=str(RUNS / "primer_tmp"), num_train_epochs=2, per_device_train_batch_size=32,
                                                learning_rate=3e-5, warmup_ratio=0.1, logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=st, args=args, train_dataset=ds, loss=MultipleNegativesRankingLoss(st)).train()
    return st[0].auto_model

def predict(net, tok, rows, bs=128):
    net.eval(); P, SL = [], []
    with torch.no_grad():
        for b, texts, enc, bio, y in batches(tok, rows, bs, False):
            li, ls = net(enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV))
            P.append(torch.softmax(li, -1).cpu())
            tags = ls.argmax(-1).cpu()
            for k, t in enumerate(texts):
                offs = [tuple(o) for o in enc["offset_mapping"][k].tolist()]
                tg = tags[k].tolist()
                idx = [j for j in range(len(offs)) if offs[j] != (0, 0)]
                start = next((j for j in idx if tg[j] in (1, 2)), None)
                if start is None: SL.append(None); continue
                end = start
                while end + 1 < len(offs) and offs[end + 1] != (0, 0) and (tg[end + 1] == 2 or offs[end + 1][0] == offs[end][1]):
                    end += 1
                a0, a1 = offs[start][0], offs[end][1]
                while a0 > 0 and t[a0 - 1] != " ": a0 -= 1          # expand to whole words
                while a1 < len(t) and t[a1] != " ": a1 += 1
                SL.append(t[a0:a1].strip())
    return torch.cat(P).numpy() if P else np.zeros((0, len(LABELS))), SL

def evaluate(P, SL, rows):
    y = np.array([L2I[r["label"]] for r in rows]); pred = P.argmax(1)
    srt = np.sort(P, 1); margin = srt[:, -1] - srt[:, -2]
    acc = float((pred == y).mean()); mf1 = float(f1_score(y, pred, average="macro", labels=sorted(set(y))))
    per_label = {l: round(float((pred[y == i] == i).mean()), 3) for l, i in L2I.items() if (y == i).any()}
    fam = defaultdict(list)
    for t, p in zip(y, pred): fam[FAMILY_OF[LABELS[t]]].append(t == p)
    fam_acc = {f: round(float(np.mean(v)), 4) for f, v in fam.items()}
    conf = Counter((LABELS[t], LABELS[p]) for t, p in zip(y, pred) if t != p)
    slot_rows = [(r, s) for r, s, p in zip(rows, SL, pred) if r["label"] in SLOT_LABELS and LABELS[p] == r["label"]]
    unseen = [(r, s) for r, s in slot_rows if r.get("source") == "handwritten_dev"]
    seen = [(r, s) for r, s in slot_rows if r.get("source") != "handwritten_dev"]
    ex = lambda rs: round(float(np.mean([norm(r["slot_text"] or "") == (s or "") for r, s in rs])), 4) if rs else None
    slot_exact = float(np.mean([norm(r["slot_text"] or "") == (s or "") for r, s in slot_rows])) if slot_rows else 0.0
    def tf1(a, b):
        a, b = (a or "").split(), (b or "").split()
        c = sum((Counter(a) & Counter(b)).values())
        return 0.0 if not c else 2 * c / (len(a) + len(b))
    slot_f1 = float(np.mean([tf1(norm(r["slot_text"] or ""), s) for r, s in slot_rows])) if slot_rows else 0.0
    hard = [{"text": r["text"], "label": r["label"], "pred": LABELS[p], "margin": round(float(m), 3)}
            for r, p, m, t in zip(rows, pred, margin, y) if p != t or m < 0.2]
    hard.sort(key=lambda h: (h["pred"] == h["label"], h["margin"]))
    return {"n": len(rows), "acc": round(acc, 4), "macro_f1": round(mf1, 4), "family_acc": fam_acc,
            "family_error": {f: round(1 - a, 4) for f, a in fam_acc.items()},
            "per_label": per_label, "top_confusions": [[t, p, c] for (t, p), c in conf.most_common(15)],
            "slot_exact": round(slot_exact, 4), "slot_token_f1": round(slot_f1, 4),
            "slot_exact_unseen_names": ex(unseen), "slot_exact_seen_names": ex(seen), "hard_examples": hard}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--primer", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(STUDENT[0], revision=STUDENT[1])
    enc = primer_encoder(*STUDENT) if a.primer else AutoModel.from_pretrained(STUDENT[0], revision=STUDENT[1])
    net = Net(enc).to(DEV)
    train = read_jsonl(GEN / "train.jsonl") + read_jsonl(GEN / "train_asr.jsonl") + read_jsonl(GEN / "silence.jsonl")
    dev_old = read_jsonl(GEN / "dev.jsonl") + read_jsonl(DEV_HAND)   # saturated; reported for comparison only
    dev = read_jsonl(DEV_HARD) or dev_old
    dev_asr = read_jsonl(DEV_HARD_ASR) if read_jsonl(DEV_HARD) else read_jsonl(GEN / "dev_asr.jsonl")
    print(f"train {len(train)}  dev {len(dev)}  dev_asr {len(dev_asr)}  device {DEV}")
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 5e-5},
                             {"params": list(net.intent.parameters()) + list(net.slot.parameters()), "lr": 1e-3}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / 32); warm = int(0.1 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    ce = nn.CrossEntropyLoss(label_smoothing=0.05); ce_slot = nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = 0
        for b, texts, e, bio, y in batches(tok, train, 32, True):
            li, ls = net(e["input_ids"].to(DEV), e["attention_mask"].to(DEV))
            loss = ce(li, y.to(DEV)) + 1.0 * ce_slot(ls.reshape(-1, 3), bio.to(DEV).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        P, SL = predict(net, tok, dev)
        print(f"epoch {ep + 1}: loss {tot:.1f}  dev acc {(P.argmax(1) == np.array([L2I[r['label']] for r in dev])).mean():.4f}  ({time.time() - t0:.0f}s)")
    out_dir = RUNS / f"round{a.round}{a.tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / "model.pt")
    P, SL = predict(net, tok, dev); rep = {"round": a.round, "primer": a.primer, "train_rows": len(train), "dev": evaluate(P, SL, dev)}
    if dev_asr:
        P2, SL2 = predict(net, tok, dev_asr); rep["dev_asr"] = evaluate(P2, SL2, dev_asr)
    if dev is not dev_old:
        P3, SL3 = predict(net, tok, dev_old); rep["dev_old"] = {k: v for k, v in evaluate(P3, SL3, dev_old).items() if k in ("n", "acc", "macro_f1")}
    rep["diversity"] = diversity.report(read_jsonl(GEN / "train.jsonl"))
    rep["family_error"] = rep["dev"]["family_error"]; rep["top_confusions"] = rep["dev"]["top_confusions"]
    rep["hard_examples"] = rep["dev"]["hard_examples"][:60] + (rep.get("dev_asr", {}).get("hard_examples", [])[:20])
    rep["diversity_worst"] = rep["diversity"]["diversity_worst"]
    raw = GEN / "checklist" / f"r{a.round}" / "raw.txt"
    if raw.exists():
        lines = [l.rstrip("\n") for l in open(raw)]
        Pc, _ = predict(net, tok, [{"text": t, "label": "unhandled"} for t in lines])
        with open(raw.parent / "preds.txt", "w") as f:
            for p in Pc: f.write(f"{LABELS[p.argmax()]} {p.max():.4f}\n")
    REPORTS.mkdir(exist_ok=True)
    json.dump(rep, open(REPORTS / f"round{a.round}{a.tag}_eval.json", "w"), indent=1)
    d = rep["dev"]
    print(f"DEV acc {d['acc']}  macroF1 {d['macro_f1']}  slot exact {d['slot_exact']} (unseen names {d['slot_exact_unseen_names']}, seen {d['slot_exact_seen_names']}) f1 {d['slot_token_f1']}")
    if "dev_asr" in rep: print(f"DEV_ASR acc {rep['dev_asr']['acc']}  macroF1 {rep['dev_asr']['macro_f1']}")
    print("family acc:", d["family_acc"]); print("top confusions:", d["top_confusions"][:8])
    print("diversity:", {k: v for k, v in rep["diversity"].items() if k != "per_label"})

if __name__ == "__main__":
    main()
