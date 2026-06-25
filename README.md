# Multiclass KnowSAM

This repository now keeps a single active path: the 3-class KnowSAM workflow for
background plus two foreground classes.

```text
0 = background
1 = foreground class 1
2 = foreground class 2
```

The old `variants/` multi-version layout has been removed. The multiclass
training and evaluation entries are now available directly from the repository
root.

## What Is Included

- KnowSAM model code based on UNet, VNet, SAM prompt modules, and fusion output.
- Dataset loaders for `train_semi`, `val`, and `test` image/mask splits.
- Multiclass training through `train_semi_SAM.py` with `--num_classes 3`.
- Multiclass prediction through `prediction_multiclass.py`.
- V100 launch wrappers in `train_v100_multiclass.sh` and `test_v100_multiclass.sh`.
- Training/evaluation monitor utilities in `utils/training_monitor.py`.
- Dataset preparation and split-management scripts in `scripts/`.

Large datasets, training outputs, model weights, and Python cache files are
intentionally excluded from Git.

## Repository Layout

```text
Model/                       KnowSAM, SAM, UNet, VNet, and prompt modules
dataloader/                  Dataset loaders, transforms, and batch sampler
utils/                       Metrics, losses, measurement, and monitor utilities
scripts/                     Dataset preparation and experiment helper scripts
train_semi_SAM.py            Main training entry reused by multiclass training
prediction_multiclass.py     Multiclass prediction/evaluation entry
train_v100_multiclass.sh     V100 training wrapper
test_v100_multiclass.sh      V100 evaluation wrapper
requirements.txt             Python dependencies
```

## Environment

Create a Python environment with PyTorch, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The scripts are written for CUDA training. CPU execution is not the target path
for full experiments.

## Data Preparation

The expected 2D dataset layout is:

```text
SampleData/<dataset_name>/
  labeled/
    image/
    mask/
  unlabeled/
    image/
  val/
    image/
    mask/
  test/
    image/
    mask/
```

Prepare the 3-class dataset from the repository root:

```bash
python scripts/prepare_260513_dataset.py \
  --multi-class \
  --output-root ./SampleData/260513_data_multiclass
```

`SampleData/260513_data_multiclass` preserves mask labels `0/1/2` for 3-class
training. Local data directories are ignored by Git. Keep medical images,
archives, and model checkpoints outside commits.

## Train

Run the default V100 multiclass profile:

```bash
bash ./train_v100_multiclass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=32 \
LABELED_BS=16 \
bash ./train_v100_multiclass.sh
```

The default V100 profile uses `BATCH_SIZE=32`, `LABELED_BS=16`,
`MAX_ITERATIONS=10000`, and `MIXED_ITERATIONS=1000`. Keep
`BATCH_SIZE - LABELED_BS >= LABELED_BS`; the mixup path assumes the unlabeled
half is at least as large as the labeled half. If memory is still underused,
try `BATCH_SIZE=40 LABELED_BS=20`. If CUDA OOM occurs, fall back to
`BATCH_SIZE=24 LABELED_BS=12`.

## Test

Run evaluation with the default checkpoint location:

```bash
bash ./test_v100_multiclass.sh
```

Or override the checkpoint and output folder:

```bash
MODEL_PATH=./Results/Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13/SGDL_best_model.pth \
SAVE_DIR=./Results/Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13/prediction_test \
bash ./test_v100_multiclass.sh
```

The prediction script reports macro-average Dice, IoU, and HD95 over foreground
classes 1 and 2, and also writes per-class metrics.

Metrics are computed as follows:

```text
class_k Dice/IoU/HD95 = metric(pred == k, gt == k), for k in {1, 2}
avg Dice/IoU/HD95 = macro-average over foreground classes 1 and 2
```

Outputs include:

```text
prediction.log
case_metrics.csv
summary.json
original/
gt_mask/
pred_mask/
gt_color/
pred_color/
overlay/
measurement_overlay/
monitor/
```

## Notes

- `sam_vit_b_01ec64.pth` and experiment checkpoints are not committed. Download
  or place them locally when needed.
- `Results/`, `SampleData/`, `data/`, and generated comparison repositories are
  ignored.
- New work should extend the root multiclass path unless there is a deliberate
  reason to create a separate repository.

## Acknowledgements

This project builds on KnowSAM and SSL4MIS, and uses SAM-related modules for
prompt-based segmentation research.
