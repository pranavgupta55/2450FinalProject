from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import shutil

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import (
    DEFAULT_ALPHA_TRADE_OBJECTIVE,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CHECKPOINT_METRIC,
    DEFAULT_INCLUDE_TICKER,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MIN_TRADES_FOR_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_THRESHOLD_METRIC,
    DEFAULT_VALIDATION_RATIO,
    PORTFOLIO_TYPE_LONG_ONLY,
    SIGNAL_REGIME_POSITIVE_ONLY,
    TRADING_MODE_LONG_ONLY,
)
from src.alpha_signal.data.multimodal import attach_weekly_text_bundle
from src.alpha_signal.data.dataset import ensure_target_columns
from src.alpha_signal.data.splitting import (
    compute_label_audit,
    load_split_artifacts,
    time_based_train_validation_split,
)
from src.alpha_signal.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)
from src.alpha_signal.evaluation.selection import (
    select_alpha_trade_threshold,
    select_classification_threshold,
)
from src.alpha_signal.evaluation.trading import simulate_alpha_trading
from src.alpha_signal.models.multimodal_attention import (
    MODALITY_TOKENS,
    build_attention_importance_frame,
    build_multimodal_model,
)
from src.alpha_signal.models.training import save_experiment_artifacts
from src.alpha_signal.text.finbert_features import require_transformers
from src.alpha_signal.utils.io import ensure_dir, write_dataframe, write_json


MARKET_FEATURE_COLUMNS = [
    "close",
    "volume",
    "vol_ma_5",
    "price_ma_5",
    "price_ma_20",
    "volatility_20",
    "price_vs_ma_5",
    "price_vs_ma_20",
    "volume_vs_ma_5",
]
EVENT_FEATURE_COLUMNS = [
    "sec_event_count",
    "finnhub_event_count",
    "yahoo_event_count",
    "has_sec_filing",
    "has_finnhub_news",
    "has_yahoo_news",
    "has_text",
]
TEMPORAL_FEATURE_COLUMNS = [
    "week_of_year",
    "month",
    "days_from_start",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a FinBERT multimodal self-attention model over text and tabular features."
    )
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-metric", type=str, default=DEFAULT_THRESHOLD_METRIC)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-loss-weight", type=float, default=0.5)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--finbert-model-name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--unfreeze-finbert", action="store_true")
    parser.add_argument("--include-ticker", action="store_true")
    parser.add_argument("--train-on-text-only", action="store_true")
    parser.add_argument("--checkpoint-metric", type=str, default=DEFAULT_CHECKPOINT_METRIC)
    parser.add_argument("--capital-per-trade", type=float, default=10_000.0)
    parser.add_argument("--alpha-trade-threshold", type=float, default=None)
    parser.add_argument("--alpha-trade-objective", type=str, default=DEFAULT_ALPHA_TRADE_OBJECTIVE)
    parser.add_argument("--min-trades-for-threshold", type=int, default=DEFAULT_MIN_TRADES_FOR_THRESHOLD)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="Print plain progress logs every N training batches. Useful in Colab subprocess output.",
    )
    return parser.parse_args()


def save_loss_plots(output_dir: str | Path, history_df: pd.DataFrame) -> list[str]:
    resolved = ensure_dir(output_dir)
    plot_cache_dir = ensure_dir(resolved / ".matplotlib")
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Saving FinBERT training plots requires `matplotlib`. Run `pip install -r requirements.txt`."
        ) from exc

    saved_paths: list[str] = []

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_total_loss"], marker="o", label="Train Total Loss")
    plt.plot(history_df["epoch"], history_df["val_total_loss"], marker="o", label="Val Total Loss")
    plt.xticks(history_df["epoch"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FinBERT Multimodal Total Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    total_path = resolved / "loss_curve_total.png"
    plt.tight_layout()
    plt.savefig(total_path, dpi=180)
    plt.close()
    saved_paths.append(str(total_path))

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_event_loss"], marker="o", label="Train Event Loss")
    plt.plot(history_df["epoch"], history_df["val_event_loss"], marker="o", label="Val Event Loss")
    plt.plot(history_df["epoch"], history_df["train_alpha_loss"], marker="o", label="Train Alpha Loss")
    plt.plot(history_df["epoch"], history_df["val_alpha_loss"], marker="o", label="Val Alpha Loss")
    plt.xticks(history_df["epoch"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FinBERT Multimodal Loss Components")
    plt.grid(alpha=0.2)
    plt.legend()
    component_path = resolved / "loss_curve_components.png"
    plt.tight_layout()
    plt.savefig(component_path, dpi=180)
    plt.close()
    saved_paths.append(str(component_path))

    return saved_paths


def save_checkpoint(
    *,
    checkpoint_dir: str | Path,
    epoch: int,
    model,
    optimizer,
    val_result: dict,
    ticker_to_id: dict[str, int],
    normalizer: FeatureNormalizer,
    config: dict,
    torch,
) -> Path:
    resolved = ensure_dir(checkpoint_dir)
    checkpoint_path = resolved / f"epoch_{epoch:02d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_result": val_result,
            "ticker_to_id": ticker_to_id,
            "normalizer": normalizer.to_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_training_history(output_dir: str | Path, history: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    resolved = ensure_dir(output_dir)
    history_df = pd.DataFrame(history)
    write_dataframe(history_df, resolved / "training_history.csv")
    write_json(resolved / "training_history.json", {"history": history})
    plot_paths = save_loss_plots(resolved, history_df)
    return history_df, plot_paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = require_transformers()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_text_encoder_max_length(tokenizer, model) -> int:
    limits: list[int] = []

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)

    encoder_config = getattr(getattr(model, "text_encoder", None), "config", None)
    encoder_limit = getattr(encoder_config, "max_position_embeddings", None)
    if isinstance(encoder_limit, int) and encoder_limit > 0:
        limits.append(encoder_limit)

    return min(limits) if limits else 512


class FeatureNormalizer:
    def __init__(self):
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "FeatureNormalizer":
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean().fillna(0.0)
        stds = numeric.std(ddof=0).fillna(1.0).replace(0.0, 1.0)
        self.mean_ = {column: float(means[column]) for column in columns}
        self.std_ = {column: float(stds[column]) for column in columns}
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        mean = pd.Series({column: self.mean_[column] for column in columns})
        std = pd.Series({column: self.std_[column] for column in columns})
        transformed = numeric.fillna(mean)
        transformed = (transformed - mean) / std
        return transformed.to_numpy(dtype=np.float32)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": self.mean_, "std": self.std_}


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


class WeeklyMultimodalDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        market_columns: list[str],
        event_columns: list[str],
        temporal_columns: list[str],
        normalizer: FeatureNormalizer,
        ticker_to_id: dict[str, int],
        label_column: str,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.market_features = normalizer.transform(self.df, market_columns)
        self.event_features = normalizer.transform(self.df, event_columns)
        self.temporal_features = normalizer.transform(self.df, temporal_columns)
        self.texts = self.df["combined_text"].fillna("").astype(str).tolist()
        self.ticker_ids = self.df["ticker"].map(lambda value: ticker_to_id.get(str(value), 0)).to_numpy(dtype=np.int64)
        self.event_labels = self.df[label_column].astype(float).to_numpy(dtype=np.float32)
        alpha_series = pd.to_numeric(self.df["future_alpha_5d"], errors="coerce")
        self.alpha_targets = alpha_series.fillna(0.0).to_numpy(dtype=np.float32)
        self.alpha_mask = alpha_series.notna().astype(np.float32).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "market_features": self.market_features[index],
            "event_features": self.event_features[index],
            "temporal_features": self.temporal_features[index],
            "ticker_id": self.ticker_ids[index],
            "event_label": self.event_labels[index],
            "alpha_target": self.alpha_targets[index],
            "alpha_mask": self.alpha_mask[index],
        }


def build_collate_fn(tokenizer, max_length: int):
    torch, _, _ = require_transformers()

    def collate(batch):
        texts = [row["text"] for row in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "market_features": torch.tensor(np.stack([row["market_features"] for row in batch]), dtype=torch.float32),
            "event_features": torch.tensor(np.stack([row["event_features"] for row in batch]), dtype=torch.float32),
            "temporal_features": torch.tensor(np.stack([row["temporal_features"] for row in batch]), dtype=torch.float32),
            "ticker_ids": torch.tensor([row["ticker_id"] for row in batch], dtype=torch.long),
            "event_label": torch.tensor([row["event_label"] for row in batch], dtype=torch.float32),
            "alpha_target": torch.tensor([row["alpha_target"] for row in batch], dtype=torch.float32),
            "alpha_mask": torch.tensor([row["alpha_mask"] for row in batch], dtype=torch.float32),
        }

    return collate


def move_batch_to_device(batch: dict, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def compute_loss(torch, nn, outputs, batch, alpha_loss_weight: float):
    bce_loss = nn.BCEWithLogitsLoss()(outputs.event_logits, batch["event_label"])
    alpha_loss_raw = nn.SmoothL1Loss(reduction="none")(outputs.alpha_score, batch["alpha_target"])
    alpha_loss = (alpha_loss_raw * batch["alpha_mask"]).sum() / batch["alpha_mask"].sum().clamp(min=1.0)
    total_loss = bce_loss + alpha_loss_weight * alpha_loss
    return total_loss, float(bce_loss.detach().cpu()), float(alpha_loss.detach().cpu())


def evaluate_model(
    *,
    torch,
    nn,
    model,
    loader,
    device,
    threshold: float,
    alpha_loss_weight: float,
    progress_label: str | None = None,
    progress_leave: bool = True,
):
    model.eval()
    all_probabilities = []
    all_alpha_scores = []
    all_event_labels = []
    all_alpha_targets = []
    attention_maps = []
    total_loss = 0.0
    total_event_loss = 0.0
    total_alpha_loss = 0.0
    total_batches = 0

    iterator = loader
    if progress_label:
        iterator = tqdm(loader, desc=progress_label, leave=progress_leave)

    with torch.no_grad():
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                market_features=batch["market_features"],
                event_features=batch["event_features"],
                temporal_features=batch["temporal_features"],
                ticker_ids=batch["ticker_ids"],
            )
            loss, event_loss, alpha_loss = compute_loss(torch, nn, outputs, batch, alpha_loss_weight)
            total_loss += float(loss.detach().cpu())
            total_event_loss += event_loss
            total_alpha_loss += alpha_loss
            total_batches += 1

            probabilities = torch.sigmoid(outputs.event_logits).detach().cpu().numpy()
            alpha_scores = outputs.alpha_score.detach().cpu().numpy()
            event_labels = batch["event_label"].detach().cpu().numpy()
            alpha_targets = batch["alpha_target"].detach().cpu().numpy()
            alpha_mask = batch["alpha_mask"].detach().cpu().numpy().astype(bool)

            all_probabilities.append(probabilities)
            all_alpha_scores.append(alpha_scores)
            all_event_labels.append(event_labels)
            all_alpha_targets.append(np.where(alpha_mask, alpha_targets, np.nan))

            if outputs.attention_map is not None:
                attention_maps.append(outputs.attention_map.detach().cpu().numpy())

    if not all_probabilities:
        return {
            "loss": 0.0,
            "event_loss": 0.0,
            "alpha_loss": 0.0,
            "metrics": {},
            "probabilities": np.asarray([], dtype=float),
            "alpha_scores": np.asarray([], dtype=float),
            "attention_maps": np.asarray([], dtype=float),
        }

    probabilities = np.concatenate(all_probabilities)
    alpha_scores = np.concatenate(all_alpha_scores)
    event_labels = np.concatenate(all_event_labels)
    alpha_targets = np.concatenate(all_alpha_targets)

    metrics = compute_binary_classification_metrics(
        y_true=event_labels,
        y_score=probabilities,
        threshold=threshold,
    )
    metrics.update(compute_regression_metrics(alpha_targets, alpha_scores))
    attention_map = np.concatenate(attention_maps, axis=0) if attention_maps else np.asarray([], dtype=float)

    return {
        "loss": float(total_loss / max(total_batches, 1)),
        "event_loss": float(total_event_loss / max(total_batches, 1)),
        "alpha_loss": float(total_alpha_loss / max(total_batches, 1)),
        "metrics": metrics,
        "probabilities": probabilities,
        "alpha_scores": alpha_scores,
        "attention_maps": attention_map,
    }


def main():
    args = parse_args()
    torch, _, _ = require_transformers()
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    set_seed(args.random_state)
    device = resolve_device(torch)
    evaluation_threshold = 0.5 if args.threshold is None else float(args.threshold)

    train_df, test_df, split_metadata = load_split_artifacts(args.split_dir)
    train_df = ensure_target_columns(train_df)
    test_df = ensure_target_columns(test_df)
    input_dir = args.input_dir or split_metadata.get("input_dir")
    if not input_dir:
        raise SystemExit("Could not determine the dataset input directory. Pass --input-dir explicitly.")

    train_with_text = attach_weekly_text_bundle(train_df, input_dir)
    test_with_text = attach_weekly_text_bundle(test_df, input_dir)
    train_with_text = ensure_target_columns(train_with_text)
    test_with_text = ensure_target_columns(test_with_text)
    inner_train_df, val_df, validation_metadata = time_based_train_validation_split(
        train_with_text,
        validation_ratio=args.validation_ratio,
    )
    if val_df.empty:
        val_df = inner_train_df.copy()
    if args.train_on_text_only:
        filtered_train_df = inner_train_df[inner_train_df["has_text"] == 1].copy()
        if not filtered_train_df.empty:
            inner_train_df = filtered_train_df

    market_columns = available_columns(train_with_text, MARKET_FEATURE_COLUMNS)
    event_columns = available_columns(train_with_text, EVENT_FEATURE_COLUMNS)
    temporal_columns = available_columns(train_with_text, TEMPORAL_FEATURE_COLUMNS)
    if not market_columns or not event_columns or not temporal_columns:
        raise SystemExit("Missing one or more required feature groups for multimodal training.")

    normalizer = FeatureNormalizer().fit(
        inner_train_df,
        market_columns + event_columns + temporal_columns,
    )
    if args.include_ticker:
        ticker_values = sorted(str(value) for value in inner_train_df["ticker"].dropna().unique())
        ticker_to_id = {"<UNK>": 0}
        ticker_to_id.update({ticker: index + 1 for index, ticker in enumerate(ticker_values)})
    else:
        ticker_to_id = {"<UNK>": 0}

    train_dataset = WeeklyMultimodalDataset(
        inner_train_df,
        market_columns=market_columns,
        event_columns=event_columns,
        temporal_columns=temporal_columns,
        normalizer=normalizer,
        ticker_to_id=ticker_to_id,
        label_column=args.label_column,
    )
    val_dataset = WeeklyMultimodalDataset(
        val_df,
        market_columns=market_columns,
        event_columns=event_columns,
        temporal_columns=temporal_columns,
        normalizer=normalizer,
        ticker_to_id=ticker_to_id,
        label_column=args.label_column,
    )
    test_dataset = WeeklyMultimodalDataset(
        test_with_text,
        market_columns=market_columns,
        event_columns=event_columns,
        temporal_columns=temporal_columns,
        normalizer=normalizer,
        ticker_to_id=ticker_to_id,
        label_column=args.label_column,
    )

    tokenizer, model = build_multimodal_model(
        ticker_vocab_size=len(ticker_to_id),
        market_dim=len(market_columns),
        event_dim=len(event_columns),
        temporal_dim=len(temporal_columns),
        finbert_model_name=args.finbert_model_name,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_finbert=not args.unfreeze_finbert,
        local_files_only=args.local_files_only,
    )
    max_supported_length = resolve_text_encoder_max_length(tokenizer, model)
    if args.max_length > max_supported_length:
        raise SystemExit(
            f"--max-length {args.max_length} exceeds the {args.finbert_model_name} "
            f"maximum sequence length of {max_supported_length}. FinBERT/BERT truncates "
            "long text; it does not compress it. Use "
            f"`--max-length {max_supported_length}` or implement chunked long-text pooling."
        )
    model.to(device)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "finbert_multimodal_attention" / args.dataset_name
    )
    output_path = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(output_path / "checkpoints")

    collate_fn = build_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        "Training setup: "
        f"device={device}, train_rows={len(train_dataset)}, val_rows={len(val_dataset)}, "
        f"test_rows={len(test_dataset)}, train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, test_batches={len(test_loader)}, "
        f"batch_size={args.batch_size}, max_length={args.max_length}, "
        f"include_ticker={args.include_ticker}, freeze_finbert={not args.unfreeze_finbert}",
        flush=True,
    )
    print(
        "Text coverage: "
        f"train={float(inner_train_df['has_text'].mean()) if len(inner_train_df) else 0.0:.4f}, "
        f"val={float(val_df['has_text'].mean()) if len(val_df) else 0.0:.4f}, "
        f"test={float(test_with_text['has_text'].mean()) if len(test_with_text) else 0.0:.4f}",
        flush=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_config = {
        "market_dim": len(market_columns),
        "event_dim": len(event_columns),
        "temporal_dim": len(temporal_columns),
        "finbert_model_name": args.finbert_model_name,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "freeze_finbert": not args.unfreeze_finbert,
        "local_files_only": args.local_files_only,
        "max_length": args.max_length,
    }

    best_checkpoint_path: Path | None = None
    best_checkpoint_epoch: int | None = None
    best_checkpoint_metric_value = float("-inf")
    checkpoint_manifest: list[dict] = []
    history = []

    for epoch in range(args.epochs):
        model.train()
        print(f"Starting epoch {epoch + 1}/{args.epochs}", flush=True)
        running_loss = 0.0
        running_event_loss = 0.0
        running_alpha_loss = 0.0
        running_batches = 0
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
            leave=True,
        )

        for batch_index, batch in enumerate(train_iterator, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                market_features=batch["market_features"],
                event_features=batch["event_features"],
                temporal_features=batch["temporal_features"],
                ticker_ids=batch["ticker_ids"],
            )
            loss, event_loss, alpha_loss = compute_loss(
                torch,
                nn,
                outputs,
                batch,
                args.alpha_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_event_loss += event_loss
            running_alpha_loss += alpha_loss
            running_batches += 1
            train_iterator.set_postfix(
                total=f"{running_loss / running_batches:.4f}",
                event=f"{running_event_loss / running_batches:.4f}",
                alpha=f"{running_alpha_loss / running_batches:.4f}",
            )
            if args.log_every_batches > 0 and (
                batch_index == 1
                or batch_index % args.log_every_batches == 0
                or batch_index == len(train_loader)
            ):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"train_total={running_loss / running_batches:.4f} "
                    f"train_event={running_event_loss / running_batches:.4f} "
                    f"train_alpha={running_alpha_loss / running_batches:.4f}",
                    flush=True,
                )

        val_result = evaluate_model(
            torch=torch,
            nn=nn,
            model=model,
            loader=val_loader,
            device=device,
            threshold=evaluation_threshold,
            alpha_loss_weight=args.alpha_loss_weight,
            progress_label=f"Epoch {epoch + 1}/{args.epochs} val",
        )
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(running_loss / max(running_batches, 1)),
            "train_total_loss": float(running_loss / max(running_batches, 1)),
            "train_event_loss": float(running_event_loss / max(running_batches, 1)),
            "train_alpha_loss": float(running_alpha_loss / max(running_batches, 1)),
            "val_loss": val_result["loss"],
            "val_total_loss": val_result["loss"],
            "val_event_loss": val_result["event_loss"],
            "val_alpha_loss": val_result["alpha_loss"],
            "val_metrics": val_result["metrics"],
        }
        history.append(epoch_record)
        save_checkpoint_this_epoch = ((epoch + 1) % 2 == 0) or ((epoch + 1) == args.epochs)
        if save_checkpoint_this_epoch:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                val_result=val_result,
                ticker_to_id=ticker_to_id,
                normalizer=normalizer,
                config=checkpoint_config,
                torch=torch,
            )
            checkpoint_manifest.append(
                {
                    "epoch": epoch + 1,
                    "path": str(checkpoint_path),
                    "val_total_loss": val_result["loss"],
                    "val_event_loss": val_result["event_loss"],
                    "val_alpha_loss": val_result["alpha_loss"],
                    "checkpoint_metric": args.checkpoint_metric,
                    "checkpoint_metric_value": val_result["metrics"].get(args.checkpoint_metric),
                }
            )
            print(
                f"Saved checkpoint for epoch {epoch + 1}: {checkpoint_path} "
                f"({args.checkpoint_metric}={val_result['metrics'].get(args.checkpoint_metric)})"
            )
            current_metric_value = val_result["metrics"].get(args.checkpoint_metric)
            if current_metric_value is None:
                current_metric_value = float("-inf")
            if float(current_metric_value) >= best_checkpoint_metric_value:
                best_checkpoint_metric_value = float(current_metric_value)
                best_checkpoint_path = checkpoint_path
                best_checkpoint_epoch = epoch + 1

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_total={epoch_record['train_total_loss']:.4f} "
            f"train_event={epoch_record['train_event_loss']:.4f} "
            f"train_alpha={epoch_record['train_alpha_loss']:.4f} "
            f"val_total={epoch_record['val_total_loss']:.4f} "
            f"val_event={epoch_record['val_event_loss']:.4f} "
            f"val_alpha={epoch_record['val_alpha_loss']:.4f}",
            flush=True,
        )

    if best_checkpoint_path is None:
        best_checkpoint_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=args.epochs,
            model=model,
            optimizer=optimizer,
            val_result=val_result,
            ticker_to_id=ticker_to_id,
            normalizer=normalizer,
            config=checkpoint_config,
            torch=torch,
        )
        best_checkpoint_epoch = args.epochs

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_val_result = evaluate_model(
        torch=torch,
        nn=nn,
        model=model,
        loader=val_loader,
        device=device,
        threshold=0.5,
        alpha_loss_weight=args.alpha_loss_weight,
        progress_label="Best checkpoint validation",
    )
    test_result = evaluate_model(
        torch=torch,
        nn=nn,
        model=model,
        loader=test_loader,
        device=device,
        threshold=0.5,
        alpha_loss_weight=args.alpha_loss_weight,
        progress_label="Test evaluation",
    )

    threshold_info = (
        {
            "threshold": float(args.threshold),
            "metric": args.threshold_metric,
            "metric_value": None,
            "metrics": compute_binary_classification_metrics(
                y_true=val_dataset.event_labels,
                y_score=best_val_result["probabilities"],
                threshold=float(args.threshold),
            ),
        }
        if args.threshold is not None
        else select_classification_threshold(
            y_true=val_dataset.event_labels,
            y_score=best_val_result["probabilities"],
            metric=args.threshold_metric,
        )
    )
    selected_threshold = float(threshold_info["threshold"])

    alpha_threshold_info = (
        {
            "threshold": float(args.alpha_trade_threshold),
            "objective": args.alpha_trade_objective,
            "objective_value": None,
            "summary": simulate_alpha_trading(
                pd.DataFrame(
                    {
                        "future_alpha_5d": val_dataset.alpha_targets,
                        "predicted_alpha_score": best_val_result["alpha_scores"],
                    }
                ),
                capital_per_trade=args.capital_per_trade,
                alpha_threshold=float(args.alpha_trade_threshold),
            )[1],
        }
        if args.alpha_trade_threshold is not None
        else select_alpha_trade_threshold(
            realized_alpha=val_dataset.alpha_targets,
            predicted_alpha_score=best_val_result["alpha_scores"],
            capital_per_trade=args.capital_per_trade,
            objective=args.alpha_trade_objective,
            min_trades=args.min_trades_for_threshold,
        )
    )
    selected_alpha_trade_threshold = float(alpha_threshold_info["threshold"])

    predictions = test_with_text[
        ["ticker", "week_start", "last_date", "future_alpha_5d", args.label_column]
    ].copy()
    predictions["predicted_probability"] = test_result["probabilities"]
    predictions["predicted_label"] = (predictions["predicted_probability"] >= selected_threshold).astype(int)
    predictions["predicted_signal"] = predictions["predicted_label"]
    predictions["predicted_alpha_score"] = test_result["alpha_scores"]
    predictions["predicted_direction"] = (predictions["predicted_alpha_score"] >= 0).astype(int)

    trade_log_df, trading_summary = simulate_alpha_trading(
        predictions,
        capital_per_trade=args.capital_per_trade,
        alpha_threshold=selected_alpha_trade_threshold,
    )
    feature_importance_df = build_attention_importance_frame(test_result["attention_maps"])
    label_audit = compute_label_audit(
        full_df=pd.concat([train_with_text, test_with_text], ignore_index=True),
        train_df=train_with_text,
        test_df=test_with_text,
        label_column=args.label_column,
    )

    metrics = compute_binary_classification_metrics(
        y_true=test_dataset.event_labels,
        y_score=test_result["probabilities"],
        threshold=selected_threshold,
    )
    metrics.update(compute_regression_metrics(test_dataset.alpha_targets, test_result["alpha_scores"]))
    metadata = {
        "model_name": "finbert_multimodal_attention",
        "dataset_name": args.dataset_name,
        "label_column": args.label_column,
        "trading_mode": TRADING_MODE_LONG_ONLY,
        "signal_regime": SIGNAL_REGIME_POSITIVE_ONLY,
        "portfolio_type": PORTFOLIO_TYPE_LONG_ONLY,
        "random_state": args.random_state,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "alpha_loss_weight": args.alpha_loss_weight,
        "threshold": selected_threshold,
        "threshold_metric": args.threshold_metric,
        "threshold_selection": threshold_info,
        "capital_per_trade": args.capital_per_trade,
        "alpha_trade_threshold": selected_alpha_trade_threshold,
        "alpha_trade_objective": args.alpha_trade_objective,
        "alpha_trade_threshold_selection": alpha_threshold_info,
        "finbert_model_name": args.finbert_model_name,
        "freeze_finbert": not args.unfreeze_finbert,
        "local_files_only": args.local_files_only,
        "include_ticker": args.include_ticker,
        "train_on_text_only": args.train_on_text_only,
        "dataset_format": "weekly_event_dataset",
        "text_bundle_sources": ["sec_filings", "finnhub_news", "yahoo_rss_news"],
        "checkpoint_metric": args.checkpoint_metric,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "train_rows": int(len(inner_train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_with_text)),
        **label_audit,
        "market_columns": market_columns,
        "event_columns": event_columns,
        "temporal_columns": temporal_columns,
        "modality_tokens": MODALITY_TOKENS,
        "ticker_vocab_size": len(ticker_to_id),
        "text_coverage_train": float(inner_train_df["has_text"].mean()) if len(inner_train_df) else 0.0,
        "text_coverage_test": float(test_with_text["has_text"].mean()) if len(test_with_text) else 0.0,
        "history": history,
        "validation_split": validation_metadata,
        "split_metadata": split_metadata,
        "normalizer": normalizer.to_dict(),
    }

    history_df, plot_paths = save_training_history(output_path, history)
    write_json(output_path / "checkpoint_manifest.json", {"checkpoints": checkpoint_manifest})

    output_path = save_experiment_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        metadata=metadata,
        predictions=predictions,
        feature_importance_df=feature_importance_df,
        trade_log_df=trade_log_df,
        trading_summary=trading_summary,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "ticker_to_id": ticker_to_id,
            "normalizer": normalizer.to_dict(),
            "config": {
                **checkpoint_config,
            },
        },
        output_path / "model.pt",
    )
    shutil.copy2(best_checkpoint_path, output_path / "best_checkpoint.pt")

    metadata["training_history_path"] = str(output_path / "training_history.csv")
    metadata["loss_plot_paths"] = plot_paths
    metadata["checkpoint_dir"] = str(checkpoint_dir)
    metadata["checkpoint_manifest_path"] = str(output_path / "checkpoint_manifest.json")
    metadata["best_checkpoint_path"] = str(output_path / "best_checkpoint.pt")
    metadata["best_checkpoint_epoch"] = best_checkpoint_epoch
    write_json(output_path / "metadata.json", metadata)

    print(f"Best checkpoint selected from epoch {best_checkpoint_epoch}: {output_path / 'best_checkpoint.pt'}")
    print(f"Saved training history to: {output_path / 'training_history.csv'}")
    print(f"Saved loss plots to: {', '.join(plot_paths)}")
    print(f"Saved FinBERT multimodal artifacts to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
