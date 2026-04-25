from pathlib import Path

import pandas as pd
from compress_pickle import load


def main():
    data_dir = Path("data/2days")
    header = pd.read_csv(data_dir / "flight_header.csv", index_col="Master Index")
    store = load(data_dir / "flight_data.pkl")
    print("Header shape:", header.shape)
    print("Columns:", list(header.columns))
    print("Label counts for before_after:")
    print(header["before_after"].value_counts(dropna=False))
    if "fold" in header.columns:
        print("Fold counts:")
        print(header["fold"].value_counts().sort_index())
    first_idx = header.index[0]
    print("First Master Index:", first_idx)
    print("First flight array shape:", store[first_idx].shape)


if __name__ == "__main__":
    main()
