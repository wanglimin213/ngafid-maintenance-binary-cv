from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import gdown
import requests
from tqdm import tqdm

ZENODO_RECORD_API = "https://zenodo.org/api/records/6624956"
ZENODO_DIRECT_URL = "https://zenodo.org/records/6624956/files/2days.tar.gz?download=1"
GOOGLE_DRIVE_URLS = {
    "2days": "https://drive.google.com/uc?id=1-2pxwiQNhFnhTg7whosQoF_yztD5jOM2",
}


def download_stream(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(output, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=output.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def get_zenodo_file_url(filename: str) -> str:
    try:
        meta = requests.get(ZENODO_RECORD_API, timeout=30).json()
        for item in meta.get("files", []):
            if item.get("key") == filename:
                return item["links"]["self"]
    except Exception:
        pass
    if filename == "2days.tar.gz":
        return ZENODO_DIRECT_URL
    raise RuntimeError(f"Could not find {filename} from Zenodo metadata")


def extract_tar_gz(archive: Path, destination: Path) -> None:
    print(f"Extracting {archive} -> {destination}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description="Download NGAFID 2days dataset")
    parser.add_argument("--dataset", default="2days", choices=["2days"])
    parser.add_argument("--source", default="zenodo", choices=["zenodo", "google_drive"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    archive = data_root / f"{args.dataset}.tar.gz"
    extracted_dir = data_root / args.dataset

    if extracted_dir.exists() and (extracted_dir / "flight_header.csv").exists():
        print(f"Dataset already extracted at {extracted_dir}")
        return

    if not archive.exists():
        if args.source == "zenodo":
            url = get_zenodo_file_url(f"{args.dataset}.tar.gz")
            download_stream(url, archive)
        else:
            data_root.mkdir(parents=True, exist_ok=True)
            gdown.download(GOOGLE_DRIVE_URLS[args.dataset], str(archive), quiet=False)
    else:
        print(f"Archive already exists: {archive}")

    if not args.no_extract:
        extract_tar_gz(archive, data_root)
        print("Done.")


if __name__ == "__main__":
    main()
