from __future__ import annotations


def require_transformers():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "FinBERT support requires `torch` and `transformers`. "
            "Run `pip install -r requirements.txt` first."
        ) from exc

    return torch, AutoModel, AutoTokenizer


def load_finbert_encoder(
    model_name: str = "ProsusAI/finbert",
    *,
    local_files_only: bool = False,
):
    torch, AutoModel, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    return torch, tokenizer, model


def mean_pool_hidden_state(last_hidden_state, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * expanded_mask
    summed = masked_hidden.sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
