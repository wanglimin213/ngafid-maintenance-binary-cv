from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from compress_pickle import load
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


@dataclass
class NGAFIDDataBundle:
    header: pd.DataFrame
    flight_store: object
    maxs: np.ndarray
    mins: np.ndarray


def load_ngafid_2days(data_dir: str | Path, channels: int = 23) -> NGAFIDDataBundle:
    """Load the extracted 2days dataset.

    Expected files after extraction:
      data/2days/flight_data.pkl
      data/2days/flight_header.csv
      data/2days/stats.csv
    """
    data_dir = Path(data_dir)
    header_path = data_dir / "flight_header.csv"
    data_path = data_dir / "flight_data.pkl"
    stats_path = data_dir / "stats.csv"

    missing = [str(p) for p in [header_path, data_path, stats_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files: " + ", ".join(missing) +
            "\nRun: python scripts/download_data.py --dataset 2days"
        )

    header = pd.read_csv(header_path, index_col="Master Index")
    flight_store = load(data_path)
    stats = pd.read_csv(stats_path)

    # The author code uses row 0 as maxs and row 1 as mins, columns 1:24 for 23 channels.
    maxs = stats.iloc[0, 1 : 1 + channels].to_numpy(dtype=np.float32)
    mins = stats.iloc[1, 1 : 1 + channels].to_numpy(dtype=np.float32)
    return NGAFIDDataBundle(header=header, flight_store=flight_store, maxs=maxs, mins=mins)


def _get_from_store(store: object, index) -> np.ndarray:
    """Read one flight array from the compressed pickle store.

    The official helper indexes the store by Master Index. This function is tolerant
    to dict/list-like stores and integer/string index variants.
    """
    candidates = [index]
    try:
        candidates.append(int(index))
    except Exception:
        pass
    candidates.append(str(index))

    for key in candidates:
        try:
            return np.asarray(store[key])
        except Exception:
            continue
    raise KeyError(f"Cannot find flight data for Master Index={index!r}")


def pad_or_truncate_last(arr: np.ndarray, max_length: int, channels: int) -> np.ndarray:
    """Keep the last max_length timesteps, matching the paper setup."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D flight array, got shape {arr.shape}")
    arr = arr[:, :channels]
    out = np.zeros((max_length, channels), dtype=np.float32)
    chunk = arr[-max_length:, :].astype(np.float32, copy=False)
    out[: chunk.shape[0], :] = chunk
    return out


def scale_with_paper_stats(x: np.ndarray, maxs: np.ndarray, mins: np.ndarray) -> np.ndarray:
    denom = maxs - mins
    denom = np.where(denom == 0, 1.0, denom)
    x = (x - mins) / denom
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32, copy=False)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _mask_fill_value(x: np.ndarray, mode: str, axis: int | None = None):
    mode = str(mode).lower()
    if mode == "zero":
        return 0.0
    if mode == "mean":
        return np.mean(x, axis=axis, keepdims=True).astype(np.float32)
    raise ValueError(f"Unsupported mask fill mode: {mode!r}; use 'zero' or 'mean'")


def apply_time_series_augmentation(
    x: np.ndarray,
    augmentation: Optional[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply conservative time-series augmentation to one scaled flight sample.

    All augmentations are intended for the training split only. The input x is
    expected to be a float32 array with shape (time, channels), usually after
    MinMax scaling.
    """
    if not augmentation or not _as_bool(augmentation.get("enabled", False)):
        return x

    x = x.astype(np.float32, copy=True)
    time_steps, channels = x.shape

    # 1) Time masking: mask a short continuous temporal segment across all sensors.
    time_mask_prob = float(augmentation.get("time_mask_prob", 0.0))
    time_mask_max_len = int(augmentation.get("time_mask_max_len", 0))
    num_time_masks = int(augmentation.get("num_time_masks", 1))
    time_mask_fill = augmentation.get("time_mask_fill", "zero")
    if time_mask_prob > 0 and time_mask_max_len > 0 and time_steps > 0:
        max_len = min(time_mask_max_len, time_steps)
        for _ in range(max(1, num_time_masks)):
            if rng.random() < time_mask_prob:
                mask_len = int(rng.integers(1, max_len + 1))
                start = int(rng.integers(0, time_steps - mask_len + 1))
                end = start + mask_len
                if str(time_mask_fill).lower() == "mean":
                    fill = _mask_fill_value(x, "mean", axis=0)
                    x[start:end, :] = fill
                else:
                    x[start:end, :] = 0.0

    # 2) Sensor masking: mask a small number of sensor channels for the whole flight.
    sensor_mask_prob = float(augmentation.get("sensor_mask_prob", 0.0))
    sensor_mask_max_channels = int(augmentation.get("sensor_mask_max_channels", 0))
    sensor_mask_fill = augmentation.get("sensor_mask_fill", "zero")
    if sensor_mask_prob > 0 and sensor_mask_max_channels > 0 and channels > 0:
        if rng.random() < sensor_mask_prob:
            max_ch = min(sensor_mask_max_channels, channels)
            n_ch = int(rng.integers(1, max_ch + 1))
            selected = rng.choice(channels, size=n_ch, replace=False)
            if str(sensor_mask_fill).lower() == "mean":
                fill = _mask_fill_value(x, "mean", axis=0)
                x[:, selected] = fill[:, selected]
            else:
                x[:, selected] = 0.0

    # 3) Small Gaussian jitter: improve robustness to small sensor noise.
    jitter_prob = float(augmentation.get("jitter_prob", 0.0))
    jitter_std = float(augmentation.get("jitter_std", 0.0))
    if jitter_prob > 0 and jitter_std > 0:
        if rng.random() < jitter_prob:
            noise = rng.normal(loc=0.0, scale=jitter_std, size=x.shape).astype(np.float32)
            x = x + noise

    if _as_bool(augmentation.get("clip_after_augmentation", True), default=True):
        x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32, copy=False)


class NGAFIDBinaryDataset(Dataset):
    def __init__(
        self,
        bundle: NGAFIDDataBundle,
        indices: Sequence,
        max_length: int = 4096,
        channels: int = 23,
        label_column: str = "before_after",
        scale_mode: str = "paper_stats",
        augmentation: Optional[dict] = None,
        random_seed: Optional[int] = None,
    ):
        self.bundle = bundle
        self.indices = list(indices)
        self.max_length = max_length
        self.channels = channels
        self.label_column = label_column
        self.scale_mode = scale_mode
        self.augmentation = augmentation if augmentation and _as_bool(augmentation.get("enabled", False)) else None
        self.rng = np.random.default_rng(random_seed)

        if label_column not in bundle.header.columns:
            raise KeyError(f"Label column {label_column!r} not found in flight_header.csv")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        arr = _get_from_store(self.bundle.flight_store, idx)
        x = pad_or_truncate_last(arr, self.max_length, self.channels)
        if self.scale_mode == "paper_stats":
            x = scale_with_paper_stats(x, self.bundle.maxs, self.bundle.mins)
        elif self.scale_mode in ("none", None):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown scale_mode: {self.scale_mode}")

        # Augmentation is applied only when this dataset is constructed with
        # augmentation enabled. Validation datasets should pass augmentation=None.
        x = apply_time_series_augmentation(x, self.augmentation, self.rng)

        y = float(self.bundle.header.loc[idx, self.label_column])
        # Input shape returned as (time, channels). The model will transpose to (channels, time).
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def get_fold_indices(
    header: pd.DataFrame,
    label_column: str = "before_after",
    fold_column: str = "fold",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List, List]]:
    """Return train/validation indices for each fold.

    Uses the official fold column if present. If not, falls back to StratifiedKFold.
    """
    if fold_column in header.columns:
        folds = []
        for fold in sorted(header[fold_column].dropna().unique()):
            val_mask = header[fold_column] == fold
            val_idx = header.index[val_mask].tolist()
            train_idx = header.index[~val_mask].tolist()
            folds.append((train_idx, val_idx))
        if len(folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds, found {len(folds)} in column {fold_column!r}")
        return folds

    if label_column not in header.columns:
        raise KeyError(f"Cannot create folds: missing label column {label_column!r}")

    y = header[label_column].astype(int).to_numpy()
    idx = np.array(header.index.tolist())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(idx[tr].tolist(), idx[va].tolist()) for tr, va in skf.split(np.zeros(len(y)), y)]
