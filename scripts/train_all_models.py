from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable or "python3"

DATASETS = [
    "top50_5yr",
    "ten_year_aapl_msft_nvda",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print or execute the full training command list for all supported datasets and model variants."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finbert-local-files-only", action="store_true")
    parser.add_argument("--finbert-epochs", type=int, default=10)
    parser.add_argument("--finbert-batch-size", type=int, default=8)
    return parser.parse_args()


def build_commands(args) -> list[list[str]]:
    commands: list[list[str]] = []
    for dataset_name in args.datasets:
        input_dir = REPO_ROOT / "data" / dataset_name
        split_dir = REPO_ROOT / "artifacts" / "splits" / dataset_name

        commands.append(
            [
                PYTHON,
                "scripts/prepare_train_test_split.py",
                "--input-dir",
                str(input_dir),
                "--dataset-name",
                dataset_name,
            ]
        )

        commands.append(
            [
                PYTHON,
                "scripts/train_random_baseline.py",
                "--split-dir",
                str(split_dir),
                "--dataset-name",
                dataset_name,
            ]
        )

        commands.append(
            [
                PYTHON,
                "scripts/build_sp500_benchmark.py",
                "--split-dir",
                str(split_dir),
                "--dataset-name",
                dataset_name,
            ]
        )

        for include_ticker in [False, True]:
            suffix = "with_ticker" if include_ticker else "no_ticker"

            rf_command = [
                PYTHON,
                "scripts/train_random_forest.py",
                "--split-dir",
                str(split_dir),
                "--dataset-name",
                dataset_name,
                "--output-dir",
                str(REPO_ROOT / "artifacts" / "experiments" / "random_forest" / f"{dataset_name}_{suffix}"),
            ]
            xgb_command = [
                PYTHON,
                "scripts/train_xgboost.py",
                "--split-dir",
                str(split_dir),
                "--dataset-name",
                dataset_name,
                "--output-dir",
                str(REPO_ROOT / "artifacts" / "experiments" / "xgboost" / f"{dataset_name}_{suffix}"),
            ]
            finbert_command = [
                PYTHON,
                "scripts/train_multimodal_attention.py",
                "--split-dir",
                str(split_dir),
                "--dataset-name",
                dataset_name,
                "--output-dir",
                str(
                    REPO_ROOT / "artifacts" / "experiments" / "finbert_multimodal_attention" / f"{dataset_name}_{suffix}"
                ),
                "--epochs",
                str(args.finbert_epochs),
                "--batch-size",
                str(args.finbert_batch_size),
            ]
            if args.finbert_local_files_only:
                finbert_command.append("--local-files-only")
            if include_ticker:
                rf_command.append("--include-ticker")
                xgb_command.append("--include-ticker")
                finbert_command.append("--include-ticker")

            commands.extend([rf_command, xgb_command, finbert_command])
    return commands


def main():
    args = parse_args()
    commands = build_commands(args)

    if not args.execute:
        for command in commands:
            print(shlex.join(command))
        return

    for command in commands:
        print(f"Running: {shlex.join(command)}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
