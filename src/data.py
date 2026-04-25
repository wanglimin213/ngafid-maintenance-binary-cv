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


class NGAFIDBinaryDataset(Dataset):
    def __init__(
        self,
        bundle: NGAFIDDataBundle,
        indices: Sequence,
        max_length: int = 4096,
        channels: int = 23,
        label_column: str = "before_after",
        scale_mode: str = "paper_stats",
    ):
        self.bundle = bundle
        self.indices = list(indices)
        self.max_length = max_length
        self.channels = channels
        self.label_column = label_column
        self.scale_mode = scale_mode

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
