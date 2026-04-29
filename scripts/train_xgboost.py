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
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost baseline.")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--capital-per-trade", type=float, default=10_000.0)
    parser.add_argument("--alpha-trade-threshold", type=float, default=0.0)
    return parser.parse_args()


def build_model(y_train, random_state: int):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise SystemExit(
            "xgboost is required. Run `pip install -r requirements.txt` first."
        ) from exc
    except Exception as exc:
        raise SystemExit(
            "xgboost is installed but could not load its native library. "
            "On macOS, install OpenMP with `brew install libomp`, then rerun this script."
        ) from exc

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, negatives / positives) if positives else 1.0

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    alpha_regressor = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    return classifier, alpha_regressor


def main():
    args = parse_args()
    train_df, test_df, split_metadata = load_split_artifacts(args.split_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "xgboost" / args.dataset_name
    )

    y_train = train_df[DEFAULT_LABEL_COLUMN].astype(int).to_numpy()
    model, alpha_regressor = build_model(y_train=y_train, random_state=args.random_state)
    results = train_and_evaluate_model(
        model=model,
        model_name="xgboost",
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

    print(f"Saved XGBoost artifacts to: {output_dir}")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
