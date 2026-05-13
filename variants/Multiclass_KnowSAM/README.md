# Multiclass KnowSAM

This variant trains the baseline KnowSAM path as a 3-class segmentation model:

```text
0 = background
1 = foreground class 1
2 = foreground class 2
```

It uses `SampleData/260513_data_multiclass`, whose masks preserve the original
integer labels from `data/260513_data/labeled/masks`.

## Prepare Data

From the repository root:

```bash
python scripts/prepare_260513_dataset.py \
  --multi-class \
  --output-root ./SampleData/260513_data_multiclass
```

## Train

```bash
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=12 \
LABELED_BS=6 \
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

## Test

```bash
bash ./variants/Multiclass_KnowSAM/test_v100_multiclass.sh
```

The prediction script reports macro-average Dice, IoU, and HD95 over foreground
classes 1 and 2, and also writes per-class metrics.
