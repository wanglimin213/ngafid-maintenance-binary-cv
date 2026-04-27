from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import NGAFIDBinaryDataset, get_fold_indices, load_ngafid_2days
from src.models import build_model
from src.seed import set_seed
from src.train import evaluate, run_one_epoch
from src.utils import ensure_dir, get_device, load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold CV for NGAFID maintenance binary detection")
    parser.add_argument("--config", type=str, default="configs/binary_inception.yaml")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    scheduler_name = train_cfg.get("scheduler", None)
    if scheduler_name is None or str(scheduler_name).lower() in {"none", "null"}:
        return None

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 5)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")



class FocalLossWithLogits(torch.nn.Module):
    """Binary focal loss implemented on logits."""

    def __init__(self, alpha: float = 0.55, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_criterion(train_cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build binary classification loss.

    Supported values:
    - bce: standard BCEWithLogitsLoss
    - weighted_bce: BCEWithLogitsLoss with positive-class pos_weight
    - focal: binary focal loss on logits
    """
    loss_name = str(train_cfg.get("loss", "bce")).lower()

    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()

    if loss_name == "weighted_bce":
        pos_weight = float(train_cfg.get("pos_weight", 1.0))
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )

    if loss_name == "focal":
        return FocalLossWithLogits(
            alpha=float(train_cfg.get("focal_alpha", 0.55)),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
        )

    raise ValueError(f"Unsupported loss function: {loss_name}")

def monitor_improved(current: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor == "val_loss":
        return current < best - min_delta
    if monitor in {"val_accuracy", "accuracy"}:
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping monitor: {monitor}")




def evaluate_binary_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate binary metrics for the positive class before-maintenance = 1.

    Returns:
        metrics: dict with loss, accuracy, precision, recall, f1, auroc, tn, fp, fn, tp
        y_true: ground-truth labels as int ndarray
        y_prob: predicted probability of positive class as float ndarray
        y_pred: thresholded predictions as int ndarray
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true_parts = []
    y_prob_parts = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            prob = torch.sigmoid(logits)

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
            y_true_parts.append(y.detach().cpu().numpy())
            y_prob_parts.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts).astype(int)
    y_prob = np.concatenate(y_prob_parts).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    metrics = {
        "loss": float(total_loss / max(total_count, 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": auroc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_true, y_prob, y_pred


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    results_dir = ensure_dir(out_cfg.get("results_dir", "results"))
    checkpoint_dir = ensure_dir(out_cfg.get("checkpoint_dir", "checkpoints"))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")

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

    all_fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        print("\n" + "=" * 80)
        print(f"Fold {fold_id + 1}/{len(folds)} | train={len(train_idx)} | val={len(val_idx)}")

        augmentation_cfg = data_cfg.get("augmentation", None)
        if augmentation_cfg and bool(augmentation_cfg.get("enabled", False)):
            print(f"Training augmentation enabled: {augmentation_cfg}")

        train_ds = NGAFIDBinaryDataset(
            bundle=bundle,
            indices=train_idx,
            max_length=int(data_cfg.get("max_length", 4096)),
            channels=int(data_cfg.get("channels", 23)),
            label_column=label_column,
            scale_mode=data_cfg.get("scale_mode", "paper_stats"),
            augmentation=augmentation_cfg,
            random_seed=seed + 1000 * fold_id,
        )
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

        generator = torch.Generator()
        generator.manual_seed(seed + fold_id)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            generator=generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(cfg["model"]).to(device)
        criterion = build_criterion(train_cfg, device)
        print(f"Loss function: {train_cfg.get('loss', 'bce')} | pos_weight={train_cfg.get('pos_weight', 'N/A')}")
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)

        best_val_acc = -math.inf
        best_epoch = 0

        patience = train_cfg.get("early_stopping_patience", None)
        early_monitor = str(train_cfg.get("early_stopping_monitor", "val_loss")).lower()
        min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
        best_monitor_score = math.inf if early_monitor == "val_loss" else -math.inf
        no_improve = 0

        max_steps = train_cfg.get("max_steps_per_epoch", None)
        if max_steps is not None:
            max_steps = int(max_steps)

        fold_rows = []
        for epoch in range(1, int(train_cfg.get("epochs", 200)) + 1):
            train_metrics = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                max_steps=max_steps,
                desc=f"fold {fold_id + 1} epoch {epoch} train",
            )
            val_metrics = evaluate(model, val_loader, criterion, device, desc=f"fold {fold_id + 1} epoch {epoch} val")

            if scheduler is not None:
                # ReduceLROnPlateau is driven by validation loss.
                scheduler.step(val_metrics.loss)

            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "learning_rate": current_lr,
            }
            fold_rows.append(row)
            print(
                f"Fold {fold_id + 1} Epoch {epoch:03d} | "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
                f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} | "
                f"lr={current_lr:.2e}"
            )

            # Keep reporting the best validation accuracy for comparability with earlier runs.
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_epoch = epoch
                if bool(train_cfg.get("save_best", True)):
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": cfg,
                            "fold": fold_id,
                            "epoch": epoch,
                            "val_accuracy": best_val_acc,
                            "val_loss": val_metrics.loss,
                        },
                        checkpoint_dir / f"fold_{fold_id}_best.pt",
                    )

            monitor_value = val_metrics.loss if early_monitor == "val_loss" else val_metrics.accuracy
            if monitor_improved(monitor_value, best_monitor_score, early_monitor, min_delta):
                best_monitor_score = monitor_value
                no_improve = 0
            else:
                no_improve += 1

            if patience is not None and no_improve >= int(patience):
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best accuracy epoch={best_epoch}; "
                    f"best {early_monitor}={best_monitor_score:.6f}"
                )
                break

        fold_log_path = results_dir / f"fold_{fold_id}_epochs.csv"
        with open(fold_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fold_rows)

        # Reload the best checkpoint and compute detailed validation metrics.
        best_checkpoint_path = checkpoint_dir / f"fold_{fold_id}_best.pt"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: {best_checkpoint_path} not found; evaluating the current model instead.")

        threshold = float(train_cfg.get("threshold", 0.5))
        fold_metrics, y_true, y_prob, y_pred = evaluate_binary_metrics(
            model, val_loader, criterion, device, threshold=threshold
        )
        fold_metrics = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            **fold_metrics,
        }

        predictions_path = results_dir / f"fold_{fold_id}_predictions.csv"
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
            f"Fold {fold_id + 1} Metrics | "
            f"Acc={fold_metrics['accuracy'] * 100:.2f}% "
            f"Precision={fold_metrics['precision'] * 100:.2f}% "
            f"Recall={fold_metrics['recall'] * 100:.2f}% "
            f"F1={fold_metrics['f1'] * 100:.2f}% "
            f"AUROC={fold_metrics['auroc']:.4f} "
            f"CM=[[{fold_metrics['tn']}, {fold_metrics['fp']}], "
            f"[{fold_metrics['fn']}, {fold_metrics['tp']}]]"
        )
        all_fold_results.append(fold_metrics)

    metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
    summary = {"folds": all_fold_results, "metrics_mean": {}, "metrics_std": {}}
    for name in metric_names:
        values = np.array([r[name] for r in all_fold_results], dtype=np.float64)
        summary["metrics_mean"][name] = float(np.nanmean(values))
        summary["metrics_std"][name] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"mean_{name}"] = summary["metrics_mean"][name]
        summary[f"std_{name}"] = summary["metrics_std"][name]
        summary[f"mean_{name}_percent"] = summary["metrics_mean"][name] * 100
        summary[f"std_{name}_percent"] = summary["metrics_std"][name] * 100

    aggregate_cm = {
        "tn": int(sum(r["tn"] for r in all_fold_results)),
        "fp": int(sum(r["fp"] for r in all_fold_results)),
        "fn": int(sum(r["fn"] for r in all_fold_results)),
        "tp": int(sum(r["tp"] for r in all_fold_results)),
    }
    summary["aggregate_confusion_matrix"] = aggregate_cm
    save_json(summary, results_dir / "summary.json")

    result_fields = [
        "fold", "best_epoch", "loss", "accuracy", "precision", "recall", "f1", "auroc",
        "tn", "fp", "fn", "tp",
        "accuracy_percent", "precision_percent", "recall_percent", "f1_percent",
    ]
    with open(results_dir / "cv_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        for r in all_fold_results:
            row = dict(r)
            row["accuracy_percent"] = r["accuracy"] * 100
            row["precision_percent"] = r["precision"] * 100
            row["recall_percent"] = r["recall"] * 100
            row["f1_percent"] = r["f1"] * 100
            writer.writerow(row)

    with open(results_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "pred_after_0", "pred_before_1"])
        writer.writerow(["true_after_0", aggregate_cm["tn"], aggregate_cm["fp"]])
        writer.writerow(["true_before_1", aggregate_cm["fn"], aggregate_cm["tp"]])

    print("\n" + "=" * 80)
    for r in all_fold_results:
        print(
            f"Fold {r['fold'] + 1}: "
            f"Acc={r['accuracy'] * 100:.2f}% | "
            f"Precision={r['precision'] * 100:.2f}% | "
            f"Recall={r['recall'] * 100:.2f}% | "
            f"F1={r['f1'] * 100:.2f}% | "
            f"AUROC={r['auroc']:.4f}"
        )
    for name in metric_names:
        mean = summary["metrics_mean"][name]
        std = summary["metrics_std"][name]
        if name == "auroc":
            print(f"Mean {name.upper()}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"Mean {name.capitalize()}: {mean * 100:.2f}% ± {std * 100:.2f}%")
    print(
        "Aggregated Confusion Matrix [[TN, FP], [FN, TP]]: "
        f"[[{aggregate_cm['tn']}, {aggregate_cm['fp']}], "
        f"[{aggregate_cm['fn']}, {aggregate_cm['tp']}]]"
    )
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
