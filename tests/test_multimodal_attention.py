from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.alpha_signal.models.multimodal_attention import (
    MODALITY_TOKENS,
    build_attention_importance_frame,
    build_multimodal_model,
)


class MultimodalAttentionTests(unittest.TestCase):
    def _write_tiny_bert(self, model_dir: Path) -> None:
        vocab_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "apple",
            "sec",
            "filing",
            "news",
            "demand",
            "margin",
            "growth",
            "risk",
            "update",
        ]
        (model_dir / "vocab.txt").write_text("\n".join(vocab_tokens), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"))
        tokenizer.save_pretrained(model_dir)

        encoder_config = BertConfig(
            vocab_size=len(vocab_tokens),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
        BertModel(encoder_config).save_pretrained(model_dir)

    def test_forward_fuses_text_market_event_temporal_and_ticker_tokens(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "tiny_bert"
            model_dir.mkdir()
            self._write_tiny_bert(model_dir)

            tokenizer, model = build_multimodal_model(
                ticker_vocab_size=3,
                market_dim=2,
                event_dim=2,
                temporal_dim=1,
                finbert_model_name=str(model_dir),
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                freeze_finbert=False,
                local_files_only=True,
            )

            encoded = tokenizer(
                ["apple sec filing growth update", "news demand margin risk"],
                padding=True,
                truncation=True,
                max_length=16,
                return_tensors="pt",
            )
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                market_features=torch.randn(2, 2),
                event_features=torch.randn(2, 2),
                temporal_features=torch.randn(2, 1),
                ticker_ids=torch.tensor([1, 2], dtype=torch.long),
            )

            expected_token_count = 1 + len(MODALITY_TOKENS)
            self.assertEqual(tuple(output.event_logits.shape), (2,))
            self.assertEqual(tuple(output.alpha_score.shape), (2,))
            self.assertEqual(tuple(output.attention_map.shape), (2, expected_token_count, expected_token_count))
            self.assertTrue(torch.isfinite(output.event_logits).all())
            self.assertTrue(torch.isfinite(output.alpha_score).all())

    def test_attention_importance_frame_uses_modality_tokens(self) -> None:
        attention_maps = torch.ones((3, 1 + len(MODALITY_TOKENS), 1 + len(MODALITY_TOKENS))).numpy()

        importance_df = build_attention_importance_frame(attention_maps)

        self.assertEqual(importance_df["feature"].tolist(), MODALITY_TOKENS)
        self.assertEqual(importance_df["importance"].tolist(), [1.0] * len(MODALITY_TOKENS))


if __name__ == "__main__":
    unittest.main()
