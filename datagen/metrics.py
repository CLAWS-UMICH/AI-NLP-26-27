"""Plan §6.3 metrics shared by final_eval (sealed test) and eval_dev (dev sets). Reads no data itself."""
import numpy as np
from sklearn.metrics import f1_score
from .config import LABELS, SLOT_LABELS
from .common import norm
from .train_eval import L2I

PAIRS = [("get_oxy_pri_storage", "get_oxy_sec_storage"), ("get_oxy_pri_pressure", "get_oxy_sec_pressure"),
         ("get_fan_pri_rpm", "get_fan_sec_rpm"), ("get_primary_battery_level", "get_secondary_battery_level"),
         ("get_scrubber_a_co2_storage", "get_scrubber_b_co2_storage"), ("get_coolant_gas_pressure", "get_coolant_liquid_pressure"),
         ("get_suit_pressure_co2", "get_helmet_pressure_co2")]

def metrics(rows, P, SL, with_slots):
    y = np.array([L2I[r["label"]] for r in rows]); pred = P.argmax(1)
    acc = float((pred == y).mean()); mf1 = float(f1_score(y, pred, average="macro", labels=sorted(set(y))))
    per = {l: round(float((pred[y == i] == i).mean()), 3) for l, i in L2I.items() if (y == i).any()}
    U = L2I["unhandled"]; cmd = y != U
    chatter_through = float((pred[~cmd] != U).mean()); cmd_rejected = float((pred[cmd] == U).mean())
    wrong_cmd = float(((pred != y) & (pred != U))[cmd].mean())
    pairs = {}
    for a, b in PAIRS:
        ia, ib = L2I[a], L2I[b]
        m = (y == ia) | (y == ib)
        swaps = ((y == ia) & (pred == ib)) | ((y == ib) & (pred == ia))
        conf_swaps = swaps & (P.max(1) >= 0.9)
        pairs[f"{a} / {b}"] = {"n": int(m.sum()), "wrong_side": round(float(swaps[m].mean()), 3), "confident_swaps": int(conf_swaps.sum())}
    out = {"n": len(rows), "acc": round(acc, 4), "macro_f1": round(mf1, 4), "worst_labels": sorted(per.items(), key=lambda kv: kv[1])[:6],
           "chatter_let_through": round(chatter_through, 4), "commands_predicted_unhandled": round(cmd_rejected, 4),
           "wrong_command_accepted": round(wrong_cmd, 4), "which_one_pairs": pairs,
           "errors": [(r["text"], r["label"], LABELS[p]) for r, p, t in zip(rows, pred, y) if p != t]}
    if with_slots:
        sr = [(r, s) for r, s, p in zip(rows, SL, pred) if r["label"] in SLOT_LABELS and LABELS[p] == r["label"]]
        out["slot_exact_unseen_names"] = round(float(np.mean([norm(r["slot_text"] or "") == (s or "") for r, s in sr])), 4)
        out["slot_misses"] = [(r["text"], r["slot_text"], s) for r, s in sr if norm(r["slot_text"] or "") != (s or "")]
    return out
