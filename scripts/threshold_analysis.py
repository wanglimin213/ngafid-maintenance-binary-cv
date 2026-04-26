from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze decision thresholds for NGAFID binary maintenance detection."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_augmented",
        help="Directory containing fold prediction CSV files.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        help="Minimum threshold to evaluate.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
        help="Maximum threshold to evaluate.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step size.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.80,
        help="Target recall for the before-maintenance positive class.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="threshold",
        help="Prefix for saved output files.",
    )
    return parser.parse_args()


def find_prediction_files(results_dir: Path) -> List[Path]:
    # Prefer predictions generated during training. If unavailable, use checkpoint-evaluation predictions.
    candidates = sorted(results_dir.glob("fold_*_predictions.csv"))
    if candidates:
        return candidates
    candidates = sorted(results_dir.glob("fold_*_predictions_from_checkpoint.csv"))
    if candidates:
        return candidates
    return []


def load_predictions(results_dir: Path) -> pd.DataFrame:
    prediction_files = find_prediction_files(results_dir)
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction CSV files found in {results_dir}.\n"
            "Expected files like fold_0_predictions.csv or fold_0_predictions_from_checkpoint.csv.\n"
            "Run one of these first:\n"
            "  python train_cv.py --config configs/binary_inception_augmented.yaml\n"
            "  set PYTHONPATH=%CD% && python scripts/evaluate_metrics.py --config configs/binary_inception_augmented.yaml"
        )

    frames = []
    for path in prediction_files:
        df = pd.read_csv(path)
        required = {"y_true", "y_prob"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        fold_number = int(path.name.split("_")[1])
        df = df.copy()
        df["fold"] = fold_number
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["y_true"] = combined["y_true"].astype(int)
    combined["y_prob"] = combined["y_prob"].astype(float)
    return combined


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, zero_division=0)),
        "auroc": float(auroc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


def add_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "false_positive_rate",
        "false_negative_rate",
    ]:
        if col in df.columns:
            df[f"{col}_percent"] = df[col] * 100
    return df


def select_recommendations(threshold_df: pd.DataFrame, target_recall: float) -> Dict[str, Dict]:
    recommendations: Dict[str, Dict] = {}

    def row_to_dict(row: pd.Series) -> Dict:
        d = row.to_dict()
        for k, v in list(d.items()):
            if isinstance(v, (np.integer, np.floating)):
                d[k] = v.item()
        return d

    recommendations["default_0_50"] = row_to_dict(
        threshold_df.iloc[(threshold_df["threshold"] - 0.50).abs().argmin()]
    )
    recommendations["max_f1"] = row_to_dict(threshold_df.loc[threshold_df["f1"].idxmax()])
    recommendations["max_f2_recall_weighted"] = row_to_dict(threshold_df.loc[threshold_df["f2"].idxmax()])
    recommendations["max_balanced_accuracy"] = row_to_dict(
        threshold_df.loc[threshold_df["balanced_accuracy"].idxmax()]
    )

    # Youden's J = sensitivity + specificity - 1.
    youden = threshold_df["recall"] + threshold_df["specificity"] - 1.0
    recommendations["max_youden_j"] = row_to_dict(threshold_df.loc[youden.idxmax()])

    feasible = threshold_df[threshold_df["recall"] >= target_recall]
    if not feasible.empty:
        # Among thresholds reaching the target recall, choose the one with highest precision.
        recommendations[f"target_recall_{target_recall:.2f}_best_precision"] = row_to_dict(
            feasible.loc[feasible["precision"].idxmax()]
        )
    else:
        recommendations[f"target_recall_{target_recall:.2f}_best_precision"] = {
            "note": f"No evaluated threshold reached recall >= {target_recall:.2f}."
        }

    return recommendations


def per_fold_metrics(predictions: pd.DataFrame, thresholds: List[Tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for name, threshold in thresholds:
        for fold, df_fold in predictions.groupby("fold"):
            y_true = df_fold["y_true"].to_numpy(dtype=int)
            y_prob = df_fold["y_prob"].to_numpy(dtype=float)
            metrics = compute_metrics(y_true, y_prob, threshold)
            metrics["setting"] = name
            metrics["fold"] = int(fold)
            rows.append(metrics)
    return pd.DataFrame(rows)


def print_metric_block(title: str, row: Dict) -> None:
    if "note" in row:
        print(f"\n{title}: {row['note']}")
        return
    print(f"\n{title}")
    print(
        f"threshold={row['threshold']:.2f} | "
        f"Acc={row['accuracy'] * 100:.2f}% | "
        f"Precision={row['precision'] * 100:.2f}% | "
        f"Recall={row['recall'] * 100:.2f}% | "
        f"F1={row['f1'] * 100:.2f}% | "
        f"F2={row['f2'] * 100:.2f}% | "
        f"AUROC={row['auroc']:.4f}"
    )
    print(f"CM=[[{row['tn']}, {row['fp']}], [{row['fn']}, {row['tp']}]]")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    predictions = load_predictions(results_dir)

    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_prob = predictions["y_prob"].to_numpy(dtype=float)

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
        dtype=float,
    )
    thresholds = np.round(thresholds, 6)

    rows = [compute_metrics(y_true, y_prob, float(t)) for t in thresholds]
    threshold_df = pd.DataFrame(rows)
    threshold_df = add_percent_columns(threshold_df)

    out_csv = results_dir / f"{args.output_prefix}_analysis.csv"
    threshold_df.to_csv(out_csv, index=False)

    recommendations = select_recommendations(threshold_df, args.target_recall)
    out_json = results_dir / f"{args.output_prefix}_recommendations.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)

    per_fold_thresholds = []
    for name, row in recommendations.items():
        if "threshold" in row:
            per_fold_thresholds.append((name, float(row["threshold"])))
    per_fold_df = per_fold_metrics(predictions, per_fold_thresholds)
    per_fold_df = add_percent_columns(per_fold_df)
    out_per_fold = results_dir / f"{args.output_prefix}_per_fold_metrics.csv"
    per_fold_df.to_csv(out_per_fold, index=False)

    print(f"Loaded {len(predictions)} predictions from: {results_dir}")
    print(f"Positive class: before maintenance (label=1)")
    print(f"AUROC is threshold-independent and remains the same across thresholds.")

    for name, row in recommendations.items():
        print_metric_block(name, row)

    print("\nSaved files:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_per_fold}")
    print("\nInterpretation tip:")
    print(
        "Lower thresholds usually increase recall and reduce false negatives, "
        "but they also increase false positives. For PHM use, compare the default 0.50 "
        "threshold with max-F1 and target-recall thresholds."
    )


if __name__ == "__main__":
    main()
