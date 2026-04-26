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

Run the baseline experiment:

```bash
python train_cv.py --config configs/binary_inception.yaml
```

Run the improved experiment without data augmentation:

```bash
python train_cv.py --config configs/binary_inception_improved.yaml
```

Run the improved experiment with conservative data augmentation:

```bash
python train_cv.py --config configs/binary_inception_augmented.yaml
```

The program reports:

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

To evaluate detailed metrics from saved checkpoints:

```bash
set PYTHONPATH=%CD% && python scripts/evaluate_metrics.py --config configs/binary_inception_augmented.yaml
```

This computes accuracy, precision, recall, F1-score, AUROC, and confusion matrices for each fold.

To reproduce the threshold analysis:

```bash
python scripts/threshold_analysis.py --results-dir results_augmented
```

To search thresholds with a different target recall:

```bash
python scripts/threshold_analysis.py --results-dir results_augmented --target-recall 0.85
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

## Experiments and Results

### Experiment 1: Baseline

Configuration:

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

### Experiment 2: Improved Model without Data Augmentation

Configuration:

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

---

### Experiment 3: Improved Model with Conservative Data Augmentation

Configuration:

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

augmentation:
  enabled: true
  time_masking: true
  sensor_masking: true
  jittering: true
```

The data augmentation is applied only to the training folds.  
Validation folds are not augmented.

The augmentation includes:

- **Time Masking**: randomly masks a short continuous temporal segment
- **Sensor Masking**: randomly masks a small number of sensor channels
- **Small Jittering**: adds small Gaussian noise to improve robustness

5-Fold Cross Validation result:

| Fold | Accuracy | Precision | Recall | F1-score | AUROC |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 76.72% | 76.67% | 75.51% | 76.09% | 0.8404 |
| Fold 2 | 76.28% | 76.74% | 76.67% | 76.71% | 0.8413 |
| Fold 3 | 76.32% | 76.00% | 75.87% | 75.93% | 0.8341 |
| Fold 4 | 78.46% | 76.54% | 79.40% | 77.94% | 0.8503 |
| Fold 5 | 76.76% | 77.12% | 72.73% | 74.86% | 0.8389 |

Final augmented result:

| Metric | Mean ± Std |
|---|---:|
| Accuracy | 76.91% ± 0.90% |
| Precision | 76.61% ± 0.40% |
| Recall | 76.04% ± 2.40% |
| F1-score | 76.31% ± 1.13% |
| AUROC | 0.8410 ± 0.0059 |

Aggregated confusion matrix across 5 folds:

|  | Predicted After | Predicted Before |
|---|---:|---:|
| True After | 4543 | 1301 |
| True Before | 1342 | 4260 |

The positive class is `before maintenance`.

The aggregated confusion matrix is represented as:

```text
[[TN, FP], [FN, TP]] = [[4543, 1301], [1342, 4260]]
```

This means that the model correctly detects 4260 before-maintenance flights and misses 1342 before-maintenance flights. It also correctly identifies 4543 after-maintenance flights and misclassifies 1301 after-maintenance flights as before-maintenance.

Compared with the improved model without data augmentation, conservative data augmentation slightly improves the mean accuracy from **76.59%** to **76.91%** and reduces the standard deviation from **1.43%** to **0.90%**. This suggests that the augmentation mainly improves cross-fold stability and robustness.

The AUROC of **0.8410 ± 0.0059** indicates that the model has stable threshold-independent discriminative ability between before-maintenance and after-maintenance flights.

---

## Threshold Analysis

The default binary classification threshold is:

```text
threshold = 0.50
```

At this threshold, the augmented model achieves the highest accuracy:

| Threshold | Accuracy | Precision | Recall | F1-score | AUROC | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 76.91% | 76.60% | 76.04% | 76.32% | 0.8406 | 1342 | 1301 |

However, in a PHM application, the positive class is `before maintenance`, and false negatives are especially important because they represent missed pre-maintenance flights.

Therefore, additional threshold analysis was performed using the saved validation predictions from all 5 folds.

### Recommended PHM Operating Threshold

The best F1-score is obtained at:

```text
threshold = 0.41
```

| Threshold | Accuracy | Precision | Recall | F1-score | F2-score | AUROC | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.41 | 75.76% | 71.01% | 85.29% | 77.50% | 81.99% | 0.8406 | 824 | 1951 |

Compared with the default threshold of 0.50, the threshold of 0.41:

- improves recall from **76.04%** to **85.29%**
- improves F1-score from **76.32%** to **77.50%**
- reduces false negatives from **1342** to **824**
- decreases accuracy slightly from **76.91%** to **75.76%**

This operating point is more suitable for PHM-oriented maintenance detection, where missing a pre-maintenance flight may be more costly than producing additional false alarms.

### Balanced Recall-Oriented Threshold

A more conservative recall-oriented threshold is:

```text
threshold = 0.46
```

This threshold achieves recall above 80% while keeping accuracy close to the default threshold.

| Threshold | Accuracy | Precision | Recall | F1-score | F2-score | AUROC | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.46 | 76.45% | 73.84% | 80.33% | 76.95% | 78.94% | 0.8406 | 1102 | 1594 |

Compared with the default threshold of 0.50, the threshold of 0.46:

- improves recall from **76.04%** to **80.33%**
- reduces false negatives from **1342** to **1102**
- decreases accuracy only slightly from **76.91%** to **76.45%**

### Threshold Summary

| Operating Point | Threshold | Accuracy | Precision | Recall | F1-score | Use Case |
|---|---:|---:|---:|---:|---:|---|
| Accuracy-oriented | 0.50 | 76.91% | 76.60% | 76.04% | 76.32% | Default evaluation |
| Balanced PHM | 0.46 | 76.45% | 73.84% | 80.33% | 76.95% | Better recall with small accuracy drop |
| Recall-oriented PHM | 0.41 | 75.76% | 71.01% | 85.29% | 77.50% | Reduces missed pre-maintenance flights |

The main reported model result remains the default-threshold 5-fold result. The threshold analysis provides alternative operating points for PHM applications where recall and false-negative reduction are more important than maximizing accuracy.

---

## Summary of Results

| Version | Main Changes | Accuracy | F1-score | AUROC |
|---|---|---:|---:|---:|
| Baseline | InceptionTime-like CNN, Adam, AvgPool | 70.39% ± 1.67% | N/A | N/A |
| Improved | AdamW, weight decay, scheduler, early stopping, AvgPool+MaxPool | 76.59% ± 1.43% | N/A | N/A |
| Augmented | Improved + Time Masking + Sensor Masking + Jittering | 76.91% ± 0.90% | 76.31% ± 1.13% | 0.8410 ± 0.0059 |
| Augmented + threshold tuning | Same model, threshold adjusted to 0.41 | 75.76% | 77.50% | 0.8406 |

The default threshold of 0.50 gives the highest accuracy.  
The threshold of 0.41 gives the best F1-score and substantially improves recall for before-maintenance flights.

---

## Improvement Strategy

The improved model introduces the following changes:

1. **AdamW + weight decay**  
   Reduces overfitting by regularizing model weights.

2. **ReduceLROnPlateau**  
   Reduces the learning rate when validation loss stops improving.

3. **Early Stopping**  
   Stops training when validation performance no longer improves.

4. **Global Average Pooling + Global Max Pooling**  
   Preserves both global trend information and local strong responses.

5. **Conservative Time-Series Data Augmentation**  
   Improves robustness by preventing the model from relying too heavily on specific time segments, sensor channels, or exact sensor values.

6. **Threshold Tuning**  
   Provides PHM-oriented operating points that reduce false negatives for before-maintenance flights.

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

The augmented PyTorch baseline achieves **76.91% ± 0.90%**, which is comparable to the binary detection performance reported by the original paper. However, because the model architecture, framework, training environment, and training budget are not identical to the original implementation, this result should be understood as a reproducible PyTorch baseline and improvement, not a strict reproduction of the paper's best model.

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
│   ├── binary_inception_improved.yaml
│   └── binary_inception_augmented.yaml
│
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── evaluate_metrics.py
│   └── threshold_analysis.py
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
├── results_augmented/
│   └── not uploaded to GitHub
│
├── checkpoints/
│   └── not uploaded to GitHub
│
├── checkpoints_improved/
│   └── not uploaded to GitHub
│
└── checkpoints_augmented/
    └── not uploaded to GitHub
```

---

## Reproducibility

To reproduce the current best experiment:

```bash
conda create -n ngafid python=3.10 -y
conda activate ngafid

pip install -r requirements.txt

python scripts/download_data.py --dataset 2days --source zenodo
python scripts/inspect_data.py

python train_cv.py --config configs/binary_inception_augmented.yaml
```

For NVIDIA GPU users:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cu126.txt
```

To evaluate detailed metrics from saved checkpoints:

```bash
set PYTHONPATH=%CD% && python scripts/evaluate_metrics.py --config configs/binary_inception_augmented.yaml
```

To reproduce threshold analysis:

```bash
python scripts/threshold_analysis.py --results-dir results_augmented
```

---

## Notes

The following files and directories are excluded from GitHub:

```text
data/
results/
results_quick/
results_improved/
results_augmented/
checkpoints/
checkpoints_improved/
checkpoints_augmented/
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
