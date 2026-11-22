"""
Shared paths and hyperparameters. Every script imports from here, so one change applies everywhere.

Paths can be overridden with env vars (handy for smoke tests on throwaway data):
    NLP_DATA_DIR=/some/dir NLP_RUNS_DIR=/other/dir python code/train.py
"""

import os
from pathlib import Path

# PATHS ----------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # NLP/
DATA = Path(os.environ.get("NLP_DATA_DIR", ROOT / "data"))
RUNS = Path(os.environ.get("NLP_RUNS_DIR", ROOT / "runs"))
INTENTS_FILE = ROOT / "intents.json"

SEED_FILE = DATA / "seed.jsonl"             # handwritten rows; split_seed.py splits them into test and the two pools
POOLS = DATA / "pools"                      # pools/train/*.jsonl and pools/val/*.jsonl: every source, one file each
TEMPLATES_DIR = DATA / "templates"          # one <intent>.txt per intent, expanded by generate.py
TRAIN_FILE = DATA / "train.jsonl"           # what train.py learns from (generate.py writes it)
VAL_FILE = DATA / "val.jsonl"               # built like train but only from val-pool seed groups (generate.py writes it)
TEST_FILE = DATA / "test.jsonl"             # handwritten + recorded, NEVER used for training or tuning
DEV_AUDIO_FILE = DATA / "dev_audio.jsonl"   # recording session A: Whisper errors, MIN_CONF; never trained on
AUDIO_DIR = DATA / "audio"                  # recorded wavs, audio/<session>/<speaker>/
PRIMER_FILE = DATA / "astronaut_primer.json"  # anchor/positive pairs for domain priming (build_primer.py writes it)
GLOSSARY_FILE = DATA / "suits_glossary.json"  # hand-written SUITS terms: normalization, jargon augmentation, primer
ACRONYMS_FILE = DATA / "nasa_acronyms_eva.json"  # EVA/suit subset of nasa/NASA-Acronyms (fetch_acronyms.py writes it)
SLOT_NAMES_FILE = DATA / "slot_names.txt"   # optional: one place name per line for slot swapping

# MODEL ----------------------------------------------------------

BASE_MODEL = os.environ.get("INTENT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = 48       # commands are 2-25 words; MiniLM wordpieces stay well under this
MIN_CONF = 0.5        # top softmax prob below this -> "unhandled"
UNHANDLED = "unhandled"
SLOT_INTENTS = {"Get_coordinates", "Set_navigation_target", "Add_waypoint", "Delete_waypoint", "Complete_task"}

# SPLIT ----------------------------------------------------------

SEED = 56
VAL_SIZE = 0.15       # carved out of train.jsonl, grouped so augmentations of one seed row stay on one side

# SEED SPLIT (split_seed.py) -------------------------------------

TEST_SHARE = 1 / 3    # of each intent's seed rows, by group
VAL_POOL_SHARE = 0.15 # of the remaining train-side groups
GROUP_SIM = 0.85      # skeletons of one intent at or above this cosine are one group

# DATA MIX (generate.py) -----------------------------------------

MIX_TARGETS = {"seed+tmpl": 0.49, "para": 0.16, "asr": 0.25, "unhandled": 0.02, "llm": 0.08}
LLM_MAX_SHARE = 0.08  # generate.py samples llm rows down to this share of the pool

# PRIMING (prime.py) ---------------------------------------------

PRIME = {
    "epochs": 3,          # primer is ~600 pairs, so one pass is only ~9 steps
    "batch": 64,          # in-batch negatives: bigger batch = harder contrastive task
    "lr": 2e-5,           # gentle, priming should add vocabulary, not wreck the pretrained geometry
    "warmup": 0.1,
    "scale": 20.0,        # cosine-sim temperature (sentence-transformers MNRL default)
}

# FINE-TUNING (train.py) -----------------------------------------

TRAIN = {
    "epochs": 10,
    "patience": 3,        # stop after this many epochs without val-loss improvement; best epoch is kept
    "batch": 32,
    "encoder_lr": 8e-5,
    "head_lr": 2e-3,      # head starts random so it needs much bigger steps than the pretrained encoder
    "weight_decay": 0.01,
    "warmup": 0.1,
    "label_smoothing": 0.1,  # keeps confidences honest so MIN_CONF -> unhandled works
    "dropout": 0.1,
    "grad_clip": 1.0,
}
