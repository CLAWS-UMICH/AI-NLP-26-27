"""Score a saved model on the dev sets (never the frozen test): the harder dev set, its Whisper transcripts,
and the old dev set for comparison. No reject threshold yet: argmax only.
Usage: python -m datagen.eval_dev --model runs/round4/model.pt"""
import argparse, json
from pathlib import Path
import torch
from .config import GEN, DEV_HAND, DEV_HARD, DEV_HARD_ASR, REPORTS, STUDENT
from .common import read_jsonl
from .train_eval import Net, predict, DEV
from .metrics import metrics

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="runs/round4/model.pt"); a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(STUDENT[0], revision=STUDENT[1])
    net = Net(AutoModel.from_pretrained(STUDENT[0], revision=STUDENT[1])); net.load_state_dict(torch.load(a.model, map_location="cpu")); net.to(DEV)
    sets = {"dev_hard": (read_jsonl(DEV_HARD), True),
            "dev_hard_asr": (read_jsonl(DEV_HARD_ASR), False),
            "dev_old": (read_jsonl(GEN / "dev.jsonl") + read_jsonl(DEV_HAND), True)}
    rep = {"model": a.model}
    for name, (rows, slots) in sets.items():
        if not rows: continue
        P, SL = predict(net, tok, rows); rep[name] = metrics(rows, P, SL, slots)
        m = rep[name]
        print(f"== {name}: n={m['n']} acc={m['acc']} macroF1={m['macro_f1']} chatter_through={m['chatter_let_through']} "
              f"cmd->unhandled={m['commands_predicted_unhandled']} wrong_cmd={m['wrong_command_accepted']}"
              + (f" slot_exact_unseen={m['slot_exact_unseen_names']}" if slots else ""))
        print("   worst labels:", m["worst_labels"])
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"dev_hard_{Path(a.model).parent.name}.json"
    json.dump(rep, open(out, "w"), indent=1); print("->", out)

if __name__ == "__main__":
    main()
