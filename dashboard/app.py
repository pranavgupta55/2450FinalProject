from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR


def _require_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "streamlit is required for the dashboard. Run `pip install -r requirements.txt`."
        ) from exc
    return st


def discover_runs(root: Path) -> list[Path]:
    return sorted(
        [
            run_dir
            for run_dir in root.glob("experiments/*/*")
            if (run_dir / "metrics.json").exists()
        ]
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    st = _require_streamlit()
    st.set_page_config(page_title="Alpha Prediction Dashboard", layout="wide")
    st.title("Alpha Prediction Dashboard")
    st.caption("Compares structured-model experiments saved under artifacts/experiments/")

    artifact_root = DEFAULT_ARTIFACT_DIR
    run_dirs = discover_runs(artifact_root)
    if not run_dirs:
        st.info("No experiment runs found yet. Train a model first.")
        return

    run_options = {
        f"{run_dir.parent.name} / {run_dir.name}": run_dir
        for run_dir in run_dirs
    }
    selected_label = st.selectbox("Experiment run", list(run_options.keys()))
    selected_run = run_options[selected_label]

    metrics = load_json(selected_run / "metrics.json")
    metadata = load_json(selected_run / "metadata.json")
    predictions_path = selected_run / "predictions.csv"
    feature_importance_path = selected_run / "feature_importance.csv"

    st.subheader("Metrics")
    metrics_df = pd.DataFrame([metrics]).T.reset_index()
    metrics_df.columns = ["metric", "value"]
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Run Metadata")
    st.json(metadata)

    if feature_importance_path.exists():
        st.subheader("Feature Importance")
        feature_importance_df = pd.read_csv(feature_importance_path)
        st.dataframe(feature_importance_df.head(25), use_container_width=True)

    if predictions_path.exists():
        st.subheader("Predictions Preview")
        predictions_df = pd.read_csv(predictions_path)
        st.dataframe(predictions_df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
