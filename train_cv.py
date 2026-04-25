from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import NGAFIDBinaryDataset, get_fold_indices, load_ngafid_2days
from src.models import build_model
from src.seed import set_seed
from src.train import evaluate, run_one_epoch
from src.utils import ensure_dir, get_device, load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold CV for NGAFID maintenance binary detection")
    parser.add_argument("--config", type=str, default="configs/binary_inception.yaml")
    return parser.parse_args()


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

        train_ds = NGAFIDBinaryDataset(
            bundle=bundle,
            indices=train_idx,
            max_length=int(data_cfg.get("max_length", 4096)),
            channels=int(data_cfg.get("channels", 23)),
            label_column=label_column,
            scale_mode=data_cfg.get("scale_mode", "paper_stats"),
        )
        val_ds = NGAFIDBinaryDataset(
            bundle=bundle,
            indices=val_idx,
            max_length=int(data_cfg.get("max_length", 4096)),
            channels=int(data_cfg.get("channels", 23)),
            label_column=label_column,
            scale_mode=data_cfg.get("scale_mode", "paper_stats"),
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
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        best_val_acc = -math.inf
        best_epoch = 0
        no_improve = 0
        patience = train_cfg.get("early_stopping_patience", None)
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

            row = {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
            }
            fold_rows.append(row)
            print(
                f"Fold {fold_id + 1} Epoch {epoch:03d} | "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
                f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
            )

            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_epoch = epoch
                no_improve = 0
                if bool(train_cfg.get("save_best", True)):
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": cfg,
                            "fold": fold_id,
                            "epoch": epoch,
                            "val_accuracy": best_val_acc,
                        },
                        checkpoint_dir / f"fold_{fold_id}_best.pt",
                    )
            else:
                no_improve += 1

            if patience is not None and no_improve >= int(patience):
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        fold_log_path = results_dir / f"fold_{fold_id}_epochs.csv"
        with open(fold_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fold_rows)

        print(f"Fold {fold_id + 1} Accuracy: {best_val_acc * 100:.2f}%")
        all_fold_results.append({"fold": fold_id, "best_epoch": best_epoch, "accuracy": best_val_acc})

    accuracies = np.array([r["accuracy"] for r in all_fold_results], dtype=np.float64)
    mean_acc = float(accuracies.mean())
    std_acc = float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0

    summary = {
        "folds": all_fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_accuracy_percent": mean_acc * 100,
        "std_accuracy_percent": std_acc * 100,
    }
    save_json(summary, results_dir / "summary.json")

    with open(results_dir / "cv_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "best_epoch", "accuracy", "accuracy_percent"])
        writer.writeheader()
        for r in all_fold_results:
            writer.writerow({**r, "accuracy_percent": r["accuracy"] * 100})

    print("\n" + "=" * 80)
    for r in all_fold_results:
        print(f"Fold {r['fold'] + 1} Accuracy: {r['accuracy'] * 100:.2f}%")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"Std Accuracy: {std_acc * 100:.2f}%")
    print(f"Mean ± Std: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
