"""
The intent model: pretrained encoder -> mean pool -> dropout -> linear head (one logit per intent).

- encoder is a Hugging Face name (MiniLM by default) or a local folder written by prime.py
- train.py applies cross-entropy to the logits, predict.py applies softmax
- for inference, IntentModel.load(run_dir) rebuilds the architecture from the run's saved config
  and loads model.pt, so it never needs the original primed folder or a download
"""

import json

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from config import MAX_TOKENS


def mean_pool(token_vecs, attention_mask):
    """Average token vectors into one sentence vector, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(token_vecs.dtype)
    return (token_vecs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class IntentModel(nn.Module):
    def __init__(self, num_intents, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, num_intents)

    @classmethod
    def from_pretrained(cls, encoder_name, num_intents, dropout=0.1):
        return cls(num_intents, AutoModel.from_pretrained(encoder_name), dropout)

    @classmethod
    def load(cls, run_dir, device="cpu"):
        """Rebuild from runs/<run>/encoder_config.json + model.pt (no pretrained download)."""
        cfg = AutoConfig.from_pretrained(run_dir / "encoder_config")
        labels = json.loads((run_dir / "label2id.json").read_text(encoding="utf-8"))
        model = cls(len(labels), AutoModel.from_config(cfg))
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        return model.to(device).eval()

    def embed(self, batch):
        return mean_pool(self.encoder(**batch).last_hidden_state, batch["attention_mask"])

    def forward(self, batch):
        return self.head(self.dropout(self.embed(batch)))


def load_tokenizer(name_or_dir):
    return AutoTokenizer.from_pretrained(name_or_dir)


def tokenize(tokenizer, texts, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
