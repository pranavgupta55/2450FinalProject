from __future__ import annotations

from pathlib import Path

from scripts.build_strategy_analysis import (
    FINBERT_STRATEGY_NAME,
    discover_benchmark_source,
    discover_model_strategy_sources,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ticker,week_start,future_alpha_5d,predicted_alpha_score\n")


def test_strategy_discovery_skips_missing_finbert_without_failing(tmp_path):
    dataset_name = "demo_ds"
    _touch(tmp_path / "random_forest" / dataset_name / "predictions.csv")
    _touch(tmp_path / "xgboost" / dataset_name / "predictions.csv")

    sources, skipped = discover_model_strategy_sources(tmp_path, dataset_name)

    assert [source.strategy_name for source in sources] == [
        "random_forest_strategy",
        "xgboost_strategy",
    ]
    assert FINBERT_STRATEGY_NAME in skipped


def test_strategy_discovery_supports_finbert_multimodal_alias(tmp_path):
    dataset_name = "demo_ds"
    _touch(tmp_path / "finbert_multimodal_attention" / dataset_name / "predictions.csv")

    sources, skipped = discover_model_strategy_sources(tmp_path, dataset_name)
    finbert_sources = [
        source for source in sources if source.strategy_name == FINBERT_STRATEGY_NAME
    ]

    assert len(finbert_sources) == 1
    assert finbert_sources[0].source_model_name == "finbert_multimodal_attention"
    assert FINBERT_STRATEGY_NAME not in skipped


def test_strategy_discovery_uses_deterministic_finbert_precedence(tmp_path):
    dataset_name = "demo_ds"
    _touch(tmp_path / "finbert_multimodal_attention" / dataset_name / "predictions.csv")
    _touch(tmp_path / "finbert" / dataset_name / "predictions.csv")

    sources, skipped = discover_model_strategy_sources(tmp_path, dataset_name)
    finbert_sources = [
        source for source in sources if source.strategy_name == FINBERT_STRATEGY_NAME
    ]

    assert len(finbert_sources) == 1
    assert finbert_sources[0].source_model_name == "finbert_multimodal_attention"
    assert f"{FINBERT_STRATEGY_NAME}:finbert" in skipped


def test_benchmark_discovery_is_optional(tmp_path):
    dataset_name = "demo_ds"

    benchmark_dir, skip_reason = discover_benchmark_source(tmp_path, dataset_name)

    assert benchmark_dir == tmp_path / "sp500_buy_hold" / dataset_name
    assert skip_reason is not None
