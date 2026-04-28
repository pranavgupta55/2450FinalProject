from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.alpha_signal.text.finbert_features import load_finbert_encoder, mean_pool_hidden_state


MODALITY_TOKENS = ["text", "market", "event", "temporal", "ticker"]


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(
            "Multimodal attention training requires `torch`. "
            "Run `pip install -r requirements.txt` first."
        ) from exc

    return torch, nn


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


@dataclass
class MultimodalForwardOutput:
    event_logits: Any
    alpha_score: Any
    attention_map: Any


def _build_fusion_block(d_model: int, num_heads: int, dropout: float):
    _, nn = _require_torch()

    class FusionTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )

        def forward(self, tokens):
            attn_output, attn_weights = self.attn(
                tokens,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            tokens = self.norm1(tokens + self.dropout(attn_output))
            tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))
            return tokens, attn_weights

    return FusionTransformerBlock()


def build_attention_importance_frame(attention_maps) -> pd.DataFrame:
    if attention_maps is None or len(attention_maps) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    import numpy as np

    attention_maps = np.asarray(attention_maps, dtype=float)
    cls_attention = attention_maps[:, 0, 1:]
    importance = cls_attention.mean(axis=0)
    feature_importance_df = pd.DataFrame(
        {
            "feature": MODALITY_TOKENS,
            "importance": importance,
        }
    )
    return feature_importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_multimodal_model(
    *,
    ticker_vocab_size: int,
    market_dim: int,
    event_dim: int,
    temporal_dim: int,
    finbert_model_name: str = "ProsusAI/finbert",
    d_model: int = 192,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    freeze_finbert: bool = True,
    local_files_only: bool = False,
):
    torch, nn = _require_torch()
    _, tokenizer, text_encoder = load_finbert_encoder(
        model_name=finbert_model_name,
        local_files_only=local_files_only,
    )

    class FinBERTMultimodalAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            if freeze_finbert:
                for parameter in self.text_encoder.parameters():
                    parameter.requires_grad = False

            text_hidden = int(self.text_encoder.config.hidden_size)
            self.text_projection = _build_mlp(text_hidden, d_model, d_model, dropout)
            self.market_projection = _build_mlp(market_dim, d_model, d_model, dropout)
            self.event_projection = _build_mlp(event_dim, d_model, d_model, dropout)
            self.temporal_projection = _build_mlp(temporal_dim, d_model, d_model, dropout)
            self.ticker_embedding = nn.Embedding(ticker_vocab_size, d_model)
            self.ticker_projection = nn.LayerNorm(d_model)

            token_count = 1 + len(MODALITY_TOKENS)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.position_embedding = nn.Parameter(torch.randn(1, token_count, d_model) * 0.02)
            self.dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList(
                [
                    _build_fusion_block(d_model=d_model, num_heads=num_heads, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
            self.event_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.alpha_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            market_features,
            event_features,
            temporal_features,
            ticker_ids,
        ):
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embedding = mean_pool_hidden_state(text_outputs.last_hidden_state, attention_mask)

            text_token = self.text_projection(text_embedding).unsqueeze(1)
            market_token = self.market_projection(market_features).unsqueeze(1)
            event_token = self.event_projection(event_features).unsqueeze(1)
            temporal_token = self.temporal_projection(temporal_features).unsqueeze(1)
            ticker_token = self.ticker_projection(self.ticker_embedding(ticker_ids)).unsqueeze(1)
            cls_token = self.cls_token.expand(input_ids.size(0), -1, -1)

            tokens = torch.cat(
                [
                    cls_token,
                    text_token,
                    market_token,
                    event_token,
                    temporal_token,
                    ticker_token,
                ],
                dim=1,
            )
            tokens = self.dropout(tokens + self.position_embedding[:, : tokens.size(1), :])

            attention_weights = []
            for layer in self.layers:
                tokens, layer_attention = layer(tokens)
                attention_weights.append(layer_attention)

            tokens = self.final_norm(tokens)
            pooled = tokens[:, 0, :]
            event_logits = self.event_head(pooled).squeeze(-1)
            alpha_score = self.alpha_head(pooled).squeeze(-1)

            attention_map = None
            if attention_weights:
                stacked = torch.stack(attention_weights, dim=0)
                attention_map = stacked.mean(dim=(0, 2))

            return MultimodalForwardOutput(
                event_logits=event_logits,
                alpha_score=alpha_score,
                attention_map=attention_map,
            )

    model = FinBERTMultimodalAttentionModel()
    return tokenizer, model
