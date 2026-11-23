#!/bin/zsh
# Post-ingest steps for one round of the loop (DATA_GENERATION.md §4, §6, §7).
# Usage: datagen/round.sh K [--primer]
set -e
K=$1; shift || true
cd "$(dirname "$0")/.."
PY=.venv/bin/python; RT=.venv-runtime/bin/python; CL=.venv-checklist/bin/python
mkdir -p data/gen/asr
# 1. rows of this round (+ dev rows in round 0) -> ASR
$PY - <<PYEOF
import json, sys
sys.path.insert(0, ".")
from datagen.common import read_jsonl, write_jsonl
k = $K
rows = [r for r in read_jsonl("data/gen/train.jsonl") if r.get("round") == k]
dev = (read_jsonl("data/gen/dev.jsonl") + read_jsonl("data/dev_hand.jsonl")) if k == 0 else []
write_jsonl(f"data/gen/asr/in_r{k}.jsonl", rows + dev)
print("asr input rows:", len(rows) + len(dev))
PYEOF
$RT datagen/asr.py data/gen/asr/in_r$K.jsonl data/gen/asr/out_r$K.jsonl --workers 3 2>&1 | grep -v -i warn | tail -1
# 2. ASR variants
$PY -m datagen.asr_post --rows data/gen/train.jsonl --asr data/gen/asr/out_r$K.jsonl --out data/gen/train_asr.jsonl --cap 0.75 2>&1 | grep -v Redirects
if [ "$K" = "0" ]; then
  $PY -m datagen.asr_post --rows data/gen/dev.jsonl data/dev_hand.jsonl --asr data/gen/asr/out_r$K.jsonl --out data/gen/dev_asr.jsonl --cap 1.0 2>&1 | grep -v Redirects
fi
# 2b. confusion-table variants (train only, capped)
$PY -m datagen.confusion_aug --cap 0.75 2>&1 | grep -v Redirects
# 3. CheckList suite (dev only), 4. train + eval, 5. score
$CL datagen/checklist_suite.py build --round $K 2>&1 | grep -v -i warn | tail -2
$PY -m datagen.train_eval --round $K "$@" 2>&1 | grep -v -E "Redirects|it/s\]|Warning"
$CL datagen/checklist_suite.py score --round $K 2>&1 | grep -v -i warn
# 6. next work order
$PY - <<PYEOF
import json
k = $K
rep = json.load(open(f"reports/round{k}_eval.json")); cl = json.load(open(f"reports/round{k}_checklist.json"))
rep["checklist"] = cl; json.dump(rep, open(f"reports/round{k}_eval.json", "w"), indent=1)
PYEOF
echo "round $K done"
