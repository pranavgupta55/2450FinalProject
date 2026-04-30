from __future__ import annotations

import argparse
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
    parser.add_argument("--k-short", type=int, default=5)
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


def build_strategy_sources(experiment_root: Path, dataset_name: str) -> dict[str, Path]:
    return {
        "random_forest": experiment_root / "random_forest" / dataset_name,
        "xgboost": experiment_root / "xgboost" / dataset_name,
        "sp500_buy_hold": experiment_root / "sp500_buy_hold" / dataset_name,
    }


def main():
    args = parse_args()
    experiment_root = (
        Path(args.experiment_root)
        if args.experiment_root
        else DEFAULT_ARTIFACT_DIR / "experiments"
    )
    sources = build_strategy_sources(experiment_root, args.dataset_name)
    output_root = Path(args.output_root) if args.output_root else None
    random_state = get_default_random_state(args.random_state)

    created_outputs: dict[str, str] = {}

    for model_name in ("random_forest", "xgboost"):
        source_dir = sources[model_name]
        predictions_path = source_dir / "predictions.csv"
        predictions_df = read_csv(predictions_path)
        artifacts = build_cross_sectional_strategy(
            predictions_df=predictions_df,
            strategy_name=f"{model_name}_strategy",
            source_model_name=model_name,
            dataset_name=args.dataset_name,
            requested_k_long=args.k_long,
            requested_k_short=args.k_short,
            initial_capital=args.initial_capital,
            adaptive=True,
            selection_mode="ranked",
            source_artifact_path=str(source_dir.resolve()),
        )
        output_dir = build_strategy_output_dir(
            strategy_name=f"{model_name}_strategy",
            dataset_name=args.dataset_name,
            output_root=output_root,
        )
        save_strategy_artifacts(output_dir, artifacts)
        created_outputs[f"{model_name}_strategy"] = str(output_dir)

    benchmark_dir = sources["sp500_buy_hold"]
    benchmark_trade_log = read_csv(benchmark_dir / "trade_log.csv")
    benchmark_metadata = read_json(benchmark_dir / "metadata.json")
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

    split_dir = Path(args.split_dir)
    test_df = read_csv(split_dir / "test.csv")
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

    print(json.dumps({"created": created_outputs}, indent=2))


if __name__ == "__main__":
    main()
