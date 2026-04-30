from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR, DEFAULT_RANDOM_STATE
from src.alpha_signal.evaluation.strategy_analysis import (
    build_buy_hold_strategy_from_trade_log,
    build_cross_sectional_strategy,
    build_strategy_output_dir,
    get_default_random_state,
    save_strategy_artifacts,
)

MODEL_STRATEGY_SOURCES = ("random_forest", "xgboost")
FINBERT_SOURCE_CANDIDATES = ("finbert_multimodal_attention", "finbert")
FINBERT_STRATEGY_NAME = "finbert_strategy"


@dataclass(frozen=True)
class ModelStrategySource:
    strategy_name: str
    source_model_name: str
    source_dir: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build plug-and-play strategy analysis artifacts from saved experiment outputs."
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--experiment-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--k-long", type=int, default=5)
    parser.add_argument(
        "--k-short",
        type=int,
        default=0,
        help="Deprecated compatibility option. Current model strategies are long-only.",
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def has_predictions(source_dir: Path) -> bool:
    return (source_dir / "predictions.csv").exists()


def discover_model_strategy_sources(
    experiment_root: Path,
    dataset_name: str,
) -> tuple[list[ModelStrategySource], dict[str, str]]:
    discovered: list[ModelStrategySource] = []
    skipped: dict[str, str] = {}

    for model_name in MODEL_STRATEGY_SOURCES:
        source_dir = experiment_root / model_name / dataset_name
        strategy_name = f"{model_name}_strategy"
        if has_predictions(source_dir):
            discovered.append(
                ModelStrategySource(
                    strategy_name=strategy_name,
                    source_model_name=model_name,
                    source_dir=source_dir,
                )
            )
        else:
            skipped[strategy_name] = f"Missing predictions.csv at {source_dir}"

    finbert_matches = [
        (model_name, experiment_root / model_name / dataset_name)
        for model_name in FINBERT_SOURCE_CANDIDATES
        if has_predictions(experiment_root / model_name / dataset_name)
    ]
    if finbert_matches:
        chosen_model_name, chosen_source_dir = finbert_matches[0]
        discovered.append(
            ModelStrategySource(
                strategy_name=FINBERT_STRATEGY_NAME,
                source_model_name=chosen_model_name,
                source_dir=chosen_source_dir,
            )
        )
        for duplicate_model_name, duplicate_source_dir in finbert_matches[1:]:
            skipped[f"{FINBERT_STRATEGY_NAME}:{duplicate_model_name}"] = (
                "Duplicate FinBERT alias skipped; "
                f"using {chosen_model_name} from {chosen_source_dir} instead of {duplicate_source_dir}"
            )
    else:
        aliases = ", ".join(FINBERT_SOURCE_CANDIDATES)
        skipped[FINBERT_STRATEGY_NAME] = (
            f"Missing FinBERT predictions.csv under supported aliases: {aliases}"
        )

    return discovered, skipped


def discover_benchmark_source(
    experiment_root: Path,
    dataset_name: str,
) -> tuple[Path | None, str | None]:
    benchmark_dir = experiment_root / "sp500_buy_hold" / dataset_name
    if (benchmark_dir / "trade_log.csv").exists():
        return benchmark_dir, None
    return benchmark_dir, f"Missing trade_log.csv at {benchmark_dir}"


def main():
    args = parse_args()
    experiment_root = (
        Path(args.experiment_root)
        if args.experiment_root
        else DEFAULT_ARTIFACT_DIR / "experiments"
    )
    output_root = Path(args.output_root) if args.output_root else None
    random_state = get_default_random_state(args.random_state)

    created_outputs: dict[str, str] = {}
    skipped_outputs: dict[str, str] = {}

    model_sources, skipped_model_outputs = discover_model_strategy_sources(
        experiment_root=experiment_root,
        dataset_name=args.dataset_name,
    )
    skipped_outputs.update(skipped_model_outputs)

    for source in model_sources:
        predictions_path = source.source_dir / "predictions.csv"
        predictions_df = read_csv(predictions_path)
        artifacts = build_cross_sectional_strategy(
            predictions_df=predictions_df,
            strategy_name=source.strategy_name,
            source_model_name=source.source_model_name,
            dataset_name=args.dataset_name,
            requested_k_long=args.k_long,
            requested_k_short=args.k_short,
            initial_capital=args.initial_capital,
            adaptive=True,
            selection_mode="ranked",
            source_artifact_path=str(source.source_dir.resolve()),
        )
        output_dir = build_strategy_output_dir(
            strategy_name=source.strategy_name,
            dataset_name=args.dataset_name,
            output_root=output_root,
        )
        save_strategy_artifacts(output_dir, artifacts)
        created_outputs[source.strategy_name] = str(output_dir)

    benchmark_dir, benchmark_skip_reason = discover_benchmark_source(
        experiment_root=experiment_root,
        dataset_name=args.dataset_name,
    )
    if benchmark_dir and benchmark_skip_reason is None:
        benchmark_trade_log = read_csv(benchmark_dir / "trade_log.csv")
        benchmark_metadata = (
            read_json(benchmark_dir / "metadata.json")
            if (benchmark_dir / "metadata.json").exists()
            else {}
        )
        benchmark_artifacts = build_buy_hold_strategy_from_trade_log(
            trade_log_df=benchmark_trade_log,
            dataset_name=args.dataset_name,
            initial_capital=float(benchmark_metadata.get("initial_capital", args.initial_capital)),
            source_artifact_path=str(benchmark_dir.resolve()),
        )
        benchmark_output_dir = build_strategy_output_dir(
            strategy_name="sp500_buy_hold",
            dataset_name=args.dataset_name,
            output_root=output_root,
        )
        save_strategy_artifacts(benchmark_output_dir, benchmark_artifacts)
        created_outputs["sp500_buy_hold"] = str(benchmark_output_dir)
    elif benchmark_skip_reason is not None:
        skipped_outputs["sp500_buy_hold"] = benchmark_skip_reason

    split_dir = Path(args.split_dir)
    test_path = split_dir / "test.csv"
    if test_path.exists():
        test_df = read_csv(test_path)
        random_artifacts = build_cross_sectional_strategy(
            predictions_df=test_df,
            strategy_name="random_strategy",
            source_model_name=None,
            dataset_name=args.dataset_name,
            requested_k_long=args.k_long,
            requested_k_short=args.k_short,
            initial_capital=args.initial_capital,
            random_state=random_state,
            adaptive=True,
            selection_mode="random",
            source_artifact_path=str(split_dir.resolve()),
        )
        random_output_dir = build_strategy_output_dir(
            strategy_name="random_strategy",
            dataset_name=args.dataset_name,
            output_root=output_root,
        )
        save_strategy_artifacts(random_output_dir, random_artifacts)
        created_outputs["random_strategy"] = str(random_output_dir)
    else:
        skipped_outputs["random_strategy"] = f"Missing test.csv at {test_path}"

    print(json.dumps({"created": created_outputs, "skipped": skipped_outputs}, indent=2))


if __name__ == "__main__":
    main()
