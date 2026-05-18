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
BATCH_SIZE=32 \
LABELED_BS=16 \
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

The default V100 profile uses `BATCH_SIZE=32`, `LABELED_BS=16`,
`MAX_ITERATIONS=10000`, and `MIXED_ITERATIONS=1000`. Keep
`BATCH_SIZE - LABELED_BS >= LABELED_BS`; the mixup path assumes the unlabeled
half is at least as large as the labeled half. If memory is still underused,
try `BATCH_SIZE=40 LABELED_BS=20`. If CUDA OOM occurs, fall back to
`BATCH_SIZE=24 LABELED_BS=12`.

## Test

```bash
bash ./variants/Multiclass_KnowSAM/test_v100_multiclass.sh
```

The prediction script reports macro-average Dice, IoU, and HD95 over foreground
classes 1 and 2, and also writes per-class metrics.

Metrics are computed as follows:

```text
class_k Dice/IoU/HD95 = metric(pred == k, gt == k), for k in {1, 2}
avg Dice/IoU/HD95 = macro-average over foreground classes 1 and 2
```

Validation logs include per-class and average metrics for SAM, SGDL, UNet, and
VNet. Test logs include per-case per-class metrics and final per-class summary
metrics for SGDL.
