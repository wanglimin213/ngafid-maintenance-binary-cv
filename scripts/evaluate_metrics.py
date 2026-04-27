from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_cv import build_criterion, evaluate_binary_metrics
from src.data import NGAFIDBinaryDataset, get_fold_indices, load_ngafid_2days
from src.models import build_model
from src.seed import set_seed
from src.utils import ensure_dir, get_device, load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved 5-fold checkpoints with detailed metrics")
    parser.add_argument("--config", type=str, default="configs/binary_inception_augmented.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    checkpoint_dir = Path(args.checkpoint_dir or out_cfg.get("checkpoint_dir", "checkpoints"))
    results_dir = ensure_dir(args.results_dir or out_cfg.get("results_dir", "results"))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    bundle = load_ngafid_2days(data_cfg["data_dir"], channels=int(data_cfg.get("channels", 23)))
    label_column = data_cfg.get("label_column", "before_after")
    fold_column = data_cfg.get("fold_column", "fold")
    folds = get_fold_indices(
        bundle.header,
        label_column=label_column,
        fold_column=fold_column,
        n_splits=int(train_cfg.get("folds", 5)),
        seed=seed,
    )

    criterion = build_criterion(train_cfg, device)
    all_fold_results = []
    for fold_id, (_, val_idx) in enumerate(folds):
        checkpoint_path = checkpoint_dir / f"fold_{fold_id}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        val_ds = NGAFIDBinaryDataset(
            bundle=bundle,
            indices=val_idx,
            max_length=int(data_cfg.get("max_length", 4096)),
            channels=int(data_cfg.get("channels", 23)),
            label_column=label_column,
            scale_mode=data_cfg.get("scale_mode", "paper_stats"),
            augmentation=None,
            random_seed=None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(cfg["model"]).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics, y_true, y_prob, y_pred = evaluate_binary_metrics(
            model, val_loader, criterion, device, threshold=float(args.threshold)
        )
        fold_result = {
            "fold": fold_id,
            "checkpoint_epoch": checkpoint.get("epoch", None),
            **metrics,
        }
        all_fold_results.append(fold_result)

        predictions_path = results_dir / f"fold_{fold_id}_predictions_from_checkpoint.csv"
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["master_index", "y_true", "y_prob", "y_pred"])
            writer.writeheader()
            for idx, yt, yp, yhat in zip(val_idx, y_true, y_prob, y_pred):
                writer.writerow({
                    "master_index": idx,
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yhat),
                })

        print(
            f"Fold {fold_id + 1}: "
            f"Acc={metrics['accuracy'] * 100:.2f}% | "
            f"Precision={metrics['precision'] * 100:.2f}% | "
            f"Recall={metrics['recall'] * 100:.2f}% | "
            f"F1={metrics['f1'] * 100:.2f}% | "
            f"AUROC={metrics['auroc']:.4f} | "
            f"CM=[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]"
        )

    metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
    summary = {"folds": all_fold_results, "metrics_mean": {}, "metrics_std": {}}
    for name in metric_names:
        values = np.array([r[name] for r in all_fold_results], dtype=np.float64)
        summary["metrics_mean"][name] = float(np.nanmean(values))
        summary["metrics_std"][name] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

    aggregate_cm = {
        "tn": int(sum(r["tn"] for r in all_fold_results)),
        "fp": int(sum(r["fp"] for r in all_fold_results)),
        "fn": int(sum(r["fn"] for r in all_fold_results)),
        "tp": int(sum(r["tp"] for r in all_fold_results)),
    }
    summary["aggregate_confusion_matrix"] = aggregate_cm
    save_json(summary, results_dir / "metrics_from_checkpoints_summary.json")

    fieldnames = [
        "fold", "checkpoint_epoch", "loss", "accuracy", "precision", "recall", "f1", "auroc",
        "tn", "fp", "fn", "tp",
        "accuracy_percent", "precision_percent", "recall_percent", "f1_percent",
    ]
    with open(results_dir / "metrics_from_checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_fold_results:
            row = dict(r)
            row["accuracy_percent"] = r["accuracy"] * 100
            row["precision_percent"] = r["precision"] * 100
            row["recall_percent"] = r["recall"] * 100
            row["f1_percent"] = r["f1"] * 100
            writer.writerow(row)

    with open(results_dir / "confusion_matrix_from_checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "pred_after_0", "pred_before_1"])
        writer.writerow(["true_after_0", aggregate_cm["tn"], aggregate_cm["fp"]])
        writer.writerow(["true_before_1", aggregate_cm["fn"], aggregate_cm["tp"]])

    print("\nSummary")
    for name in metric_names:
        mean = summary["metrics_mean"][name]
        std = summary["metrics_std"][name]
        if name == "auroc":
            print(f"Mean {name.upper()}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"Mean {name.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    print(
        "Aggregated Confusion Matrix [[TN, FP], [FN, TP]]: "
        f"[[{aggregate_cm['tn']}, {aggregate_cm['fp']}], [{aggregate_cm['fn']}, {aggregate_cm['tp']}]]"
    )
    print(f"Saved detailed metrics to: {results_dir}")


if __name__ == "__main__":
    main()
