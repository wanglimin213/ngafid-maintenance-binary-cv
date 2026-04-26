# NGAFID Maintenance Binary Detection

This project reproduces and extends the **binary maintenance issue detection** task on the NGAFID Aviation Maintenance Dataset.

The task is to classify each flight as either:

- `0`: after maintenance flight
- `1`: before maintenance flight

Each input sample is a multivariate time series:

```text
4096 time steps × 23 sensor channels
```

The project implements a reproducible **5-Fold Cross Validation** training pipeline and reports the accuracy of each fold as well as the mean and standard deviation.

---

## Task Description

The binary classification task follows the maintenance issue detection setting from the NGAFID Aviation Maintenance Dataset paper.

The model receives one flight record as input and predicts whether the flight occurred before or after a maintenance event.

```text
Input:  flight sensor time series, shape = 4096 × 23
Output: binary label, 0 = after maintenance, 1 = before maintenance
```

---

## Dataset

This project uses the released `2days` benchmark subset from the NGAFID Aviation Maintenance Dataset.

The dataset is **not included** in this repository because of its large size.

To download the dataset:

```bash
python scripts/download_data.py --dataset 2days --source zenodo
```

Then inspect the dataset:

```bash
python scripts/inspect_data.py
```

Expected data directory:

```text
data/2days/
├── flight_header.csv
├── flight_data.pkl
└── stats.csv
```

---

## Environment Setup

Create a clean conda environment:

```bash
conda create -n ngafid python=3.10 -y
conda activate ngafid
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU users, install CUDA-enabled PyTorch:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cu126.txt
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Quick Test

Run a quick test to check whether the data loading, model construction, and 5-fold training pipeline work correctly:

```bash
python train_cv.py --config configs/quick_test.yaml
```

This quick test is only for debugging.  
Its accuracy is not intended to be reported as a formal result.

---

## Main Training

Run the baseline 5-Fold Cross Validation experiment:

```bash
python train_cv.py --config configs/binary_inception.yaml
```

Run the improved experiment:

```bash
python train_cv.py --config configs/binary_inception_improved.yaml
```

The program will report:

```text
Fold 1 Accuracy
Fold 2 Accuracy
Fold 3 Accuracy
Fold 4 Accuracy
Fold 5 Accuracy
Mean Accuracy
Std Accuracy
Mean ± Std
```

---

## Model

This project uses a PyTorch implementation of a lightweight **InceptionTime-like 1D-CNN** binary classifier.

The model contains:

- multi-scale 1D convolution branches
- residual connections
- batch normalization
- ReLU activation
- global average pooling or average-max pooling
- a final binary classification layer

The model takes input with shape:

```text
batch_size × 4096 × 23
```

Before entering the convolutional layers, the input is transposed to:

```text
batch_size × 23 × 4096
```

This format is required by PyTorch `Conv1d`.

---

## Baseline Result

Baseline configuration:

```yaml
device: cuda
epochs: 100
max_steps_per_epoch: 100
batch_size: 16
optimizer: Adam
pooling: global average pooling
```

5-Fold Cross Validation result:

| Fold | Accuracy |
|---|---:|
| Fold 1 | 69.74% |
| Fold 2 | 68.63% |
| Fold 3 | 69.46% |
| Fold 4 | 71.30% |
| Fold 5 | 72.83% |

Final baseline result:

```text
Mean Accuracy: 70.39%
Std Accuracy: 1.67%
Mean ± Std: 70.39% ± 1.67%
```

---

## Improved Result

Improved configuration:

```yaml
device: cuda
epochs: 150
max_steps_per_epoch: 100
batch_size: 24

optimizer: AdamW
weight_decay: 0.0001

scheduler: ReduceLROnPlateau
early_stopping: true

pooling: global average pooling + global max pooling
```

5-Fold Cross Validation result:

| Fold | Accuracy |
|---|---:|
| Fold 1 | 76.81% |
| Fold 2 | 76.19% |
| Fold 3 | 74.57% |
| Fold 4 | 76.80% |
| Fold 5 | 78.55% |

Final improved result:

```text
Mean Accuracy: 76.59%
Std Accuracy: 1.43%
Mean ± Std: 76.59% ± 1.43%
```

Compared with the baseline result, the improved model increases the mean accuracy from **70.39%** to **76.59%**, with a gain of **6.20 percentage points**.

---

## Improvement Strategy

The improved model introduces four changes:

1. **AdamW + weight decay**  
   Reduces overfitting by regularizing model weights.

2. **ReduceLROnPlateau**  
   Reduces the learning rate when validation loss stops improving.

3. **Early Stopping**  
   Stops training when validation performance no longer improves.

4. **Global Average Pooling + Global Max Pooling**  
   Preserves both global trend information and local strong responses, reducing the risk that short maintenance-related signals are diluted by average pooling.

---

## Difference from the Paper

This project follows the same binary maintenance detection task and uses the released `2days` benchmark subset.

However, this is **not an exact reimplementation** of the authors' original TensorFlow models.

Main differences:

| Item | Paper | This Project |
|---|---|---|
| Dataset | 2-day benchmark subset | Released 2days benchmark subset |
| Cross Validation | Preset 5-fold split | Preset fold column is used when available |
| Input length | 4096 time steps | 4096 time steps |
| Sensor channels | 23 | 23 |
| Main models | ConvMHSA, InceptionTime, MiniRocket | PyTorch InceptionTime-like 1D-CNN |
| Framework | TensorFlow / Colab | PyTorch / local GPU |
| Reported result | Mean validation accuracy | Per-fold accuracy and mean ± std |

The improved PyTorch baseline achieves **76.59% ± 1.43%**, which is comparable to the binary detection performance reported by the original paper. However, because the model architecture, framework, training environment, and training budget are not identical to the original implementation, this result should be understood as a reproducible PyTorch baseline and improvement, not a strict reproduction of the paper's best model.

---

## Project Structure

```text
ngafid-maintenance-binary-cv/
│
├── README.md
├── requirements.txt
├── requirements-cu126.txt
├── train_cv.py
├── run.sh
├── .gitignore
│
├── configs/
│   ├── quick_test.yaml
│   ├── binary_inception.yaml
│   └── binary_inception_improved.yaml
│
├── scripts/
│   ├── download_data.py
│   └── inspect_data.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── seed.py
│   ├── train.py
│   └── utils.py
│
├── data/
│   └── not uploaded to GitHub
│
├── results/
│   └── not uploaded to GitHub
│
├── results_improved/
│   └── not uploaded to GitHub
│
├── checkpoints/
│   └── not uploaded to GitHub
│
└── checkpoints_improved/
    └── not uploaded to GitHub
```

---

## Reproducibility

To reproduce the current experiment:

```bash
conda create -n ngafid python=3.10 -y
conda activate ngafid

pip install -r requirements.txt

python scripts/download_data.py --dataset 2days --source zenodo
python scripts/inspect_data.py

python train_cv.py --config configs/binary_inception_improved.yaml
```

For NVIDIA GPU users:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cu126.txt
```

---

## Notes

The following files and directories are excluded from GitHub:

```text
data/
results/
results_quick/
results_improved/
checkpoints/
checkpoints_improved/
*.pkl
*.pt
*.pth
*.tar.gz
*.zip
```

The dataset, trained model checkpoints, and result files should be regenerated locally.

---

## License

This repository is for academic reproduction and research purposes.

Please refer to the original NGAFID Aviation Maintenance Dataset release for dataset licensing information.
