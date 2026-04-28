from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR, DEFAULT_LABEL_COLUMN, DEFAULT_RANDOM_STATE
from src.alpha_signal.data.splitting import load_split_artifacts
from src.alpha_signal.models.training import train_and_evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest baseline.")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--capital-per-trade", type=float, default=10_000.0)
    parser.add_argument("--alpha-trade-threshold", type=float, default=0.0)
    return parser.parse_args()


def build_model(random_state: int):
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required. Run `pip install -r requirements.txt` first."
        ) from exc

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    alpha_regressor = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )
    return classifier, alpha_regressor


def main():
    args = parse_args()
    train_df, test_df, split_metadata = load_split_artifacts(args.split_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "random_forest" / args.dataset_name
    )

    model, alpha_regressor = build_model(args.random_state)
    results = train_and_evaluate_model(
        model=model,
        model_name="random_forest",
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        label_column=DEFAULT_LABEL_COLUMN,
        threshold=args.threshold,
        use_smote=args.use_smote,
        random_state=args.random_state,
        extra_metadata={"split_metadata": split_metadata},
        alpha_regressor=alpha_regressor,
        capital_per_trade=args.capital_per_trade,
        alpha_trade_threshold=args.alpha_trade_threshold,
    )

    print(f"Saved Random Forest artifacts to: {output_dir}")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
