"""One-time read of the frozen test set (DATA_GENERATION.md: 'Then score the frozen test set once').
Model is fixed before this runs (runs/round4/model.pt). Reports plan §6.3 metrics on test text and on
test_audio_synth (Whisper transcripts of every test row). No reject threshold yet: argmax only.
Usage: python -m datagen.final_eval --model runs/round4/model.pt"""
import argparse, json
import torch
from .config import FROZEN_TEST, REPORTS, STUDENT
from .common import read_jsonl
from .train_eval import Net, predict, DEV
from .metrics import metrics

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="runs/round4/model.pt"); a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(STUDENT[0], revision=STUDENT[1])
    net = Net(AutoModel.from_pretrained(STUDENT[0], revision=STUDENT[1])); net.load_state_dict(torch.load(a.model, map_location="cpu")); net.to(DEV)
    test = read_jsonl(FROZEN_TEST)
    P, SL = predict(net, tok, test); rep = {"model": a.model, "test_text": metrics(test, P, SL, True)}
    asr = {o["id"]: o for o in read_jsonl(FROZEN_TEST.parent / "test_asr.jsonl") if "asr" in o}
    audio = [dict(r, text=asr[r["id"]]["asr"]) for r in test if r["id"] in asr]
    P2, SL2 = predict(net, tok, audio); rep["test_audio_synth"] = metrics(audio, P2, SL2, False)
    # aggregates only: test row texts never leave data/frozen/ (they would leak into the loop)
    for k in ("test_text", "test_audio_synth"):
        rep[k].pop("errors", None); rep[k].pop("slot_misses", None)
    REPORTS.mkdir(exist_ok=True); json.dump(rep, open(REPORTS / "final_test.json", "w"), indent=1)
    for k in ("test_text", "test_audio_synth"):
        m = rep[k]
        print(f"== {k}: n={m['n']} acc={m['acc']} macroF1={m['macro_f1']} chatter_through={m['chatter_let_through']} "
              f"cmd->unhandled={m['commands_predicted_unhandled']} wrong_cmd={m['wrong_command_accepted']}")
        print("   worst labels:", m["worst_labels"])
        print("   pairs:", {k2.split(' / ')[0][4:]: (v['wrong_side'], v['confident_swaps']) for k2, v in m["which_one_pairs"].items()})
    print("   slot exact (unseen names):", rep["test_text"]["slot_exact_unseen_names"])
    print("slot misses:", rep["test_text"]["slot_misses"])

if __name__ == "__main__":
    main()
