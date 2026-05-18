# Batch A3 Inference

This folder contains a standalone batch inference script for raw A3 image folders.

Expected default input layout:

```text
input_root/
  patient_001/
    A3_001.png
    A3_002.png
  patient_002/
    A3_001.png
```

By default, outputs are written beside each source image:

```text
patient_001/
  A3_001_pred_mask.png
  A3_001_pred_color.png
  A3_001_overlay.png
```

Run KnowSAM/SGDL inference:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --model-path ".\Results\train_260513_data_label1_v100_semi_106_117_13_13\SGDL_best_model.pth"
```

Write outputs to a separate root while preserving patient folders:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --output-root "D:\A3_predictions" ^
  --model-path ".\Results\train_260513_data_label1_v100_semi_106_117_13_13\SGDL_best_model.pth"
```

Run A3-PASS inference:

```bash
python batch_inference/batch_infer_a3.py ^
  --variant a3_pass ^
  --input-root "D:\A3_images" ^
  --model-path ".\Results\A3_PASS_KnowSAM_V100_label1_106_117_13_13\fold_0\PASS_best_model.pth"
```

Useful options:

- `--include-keyword A3`: only process filenames containing `A3`.
- `--device cpu` or `--device cuda:0`: override automatic device selection.
- `--save-prob`: also save foreground probability PNG files.
- `--overwrite`: overwrite existing `*_pred_mask.png` and `*_overlay.png`.
