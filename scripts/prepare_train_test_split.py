from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR, DEFAULT_TEST_RATIO
from src.alpha_signal.data.dataset import (
    build_structured_modeling_dataset,
    get_feature_spec,
    load_weekly_event_dataset,
)
from src.alpha_signal.data.splitting import save_split_artifacts, time_based_train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a chronological train/test split from weekly_event_dataset."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or Path(args.input_dir).name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "splits" / dataset_name
    )

    raw_df = load_weekly_event_dataset(args.input_dir)
    modeling_df = build_structured_modeling_dataset(raw_df)
    train_df, test_df, metadata = time_based_train_test_split(
        modeling_df,
        test_ratio=args.test_ratio,
    )

    feature_spec = get_feature_spec(modeling_df)
    metadata.update(
        {
            "dataset_name": dataset_name,
            "input_dir": str(Path(args.input_dir).resolve()),
            "total_rows": int(len(modeling_df)),
            "feature_spec": feature_spec,
            "label_positive_rate": float(modeling_df["label_abs_alpha_gt_1pct"].mean()),
        }
    )

    save_split_artifacts(train_df, test_df, metadata, output_dir)

    print(f"Saved split artifacts to: {output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
