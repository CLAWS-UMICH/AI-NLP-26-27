"""Blind relabel (DATA_GENERATION.md §3): a different model labels each row from the guide alone,
never seeing the intended label. Uses qwen3:8b in Ollama with the answer constrained to the 36 labels."""
import json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from .config import ROOT, LABELS

MODEL = "qwen3:8b"          # first blind labeller
SECOND = "qwen3:14b"        # second blind look at disputed rows (replaces the human review queue)
_guide = (ROOT / "LABELING_GUIDE.md").read_text()
_spec = _guide[_guide.index("## 1."):_guide.index("## 5.")]  # guide §1-§4 only
SYSTEM = (
    "You label one utterance from an astronaut's headset for a spacesuit HUD. The text is a speech-to-text "
    "transcript and may contain recognition errors (e.g. 'at' for 'add', 'o two' for O2). Pick exactly one label "
    "using these rules.\n\n" + _spec +
    "\n\nAnswer as JSON: {\"label\": \"<one of the labels>\"}."
)
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}}, "required": ["label"]}

def label_one(text, timeout=180, model=None):
    body = {"model": model or MODEL, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    return json.loads(out)["label"]

def label_many(texts, workers=4, model=None):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: label_one(t, model=model), texts))

def blind_check(rows):
    """Two-step blind relabel. Returns (kept, disputed). A row is kept if qwen3:8b gives its label, or, failing that,
    if qwen3:14b does. Neither model sees the intended label."""
    p1 = label_many([r["text"] for r in rows], workers=4, model=MODEL)
    kept, second = [], []
    for r, p in zip(rows, p1):
        r["relabel_8b"] = p
        (kept if p == r["label"] else second).append(r)
    disputed = []
    if second:
        p2 = label_many([r["text"] for r in second], workers=2, model=SECOND)
        for r, p in zip(second, p2):
            r["relabel_14b"] = p
            (kept if p == r["label"] else disputed).append(r)
    return kept, disputed
