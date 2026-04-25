#!/usr/bin/env bash
set -e

python scripts/download_data.py --dataset 2days --source zenodo
python train_cv.py --config configs/binary_inception.yaml
