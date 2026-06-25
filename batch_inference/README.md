# Batch Multiclass KnowSAM Inference

This folder contains a standalone batch inference script for raw ultrasound
image folders using the current multiclass KnowSAM checkpoint.

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
  A3_001_measurement.png
```

Run multiclass KnowSAM/SGDL inference:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --model-path ".\Results\Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13\SGDL_best_model.pth"
```

Write outputs to a separate root while preserving patient folders:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --output-root "D:\A3_predictions" ^
  --model-path ".\Results\Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13\SGDL_best_model.pth"
```

Useful options:

- `--include-keyword A3`: only process filenames containing `A3`.
- `--device cpu` or `--device cuda:0`: override automatic device selection.
- `--num-classes 3`: run the default 3-class output head.
- `--save-prob`: also save foreground probability PNG files.
- `--pixel-spacing 0.12` or `--pixel-spacing 0.12,0.12`: also report width/depth in mm.
- `--disable-measurement`: skip fissure width/depth measurement and `*_measurement.png`.
- `--overwrite`: overwrite existing `*_pred_mask.png`, `*_overlay.png`, and `*_measurement.png`.

The measurement overlay draws dashed lines for:

- `width`: the opening distance between the two lips of the lateral fissure.
- `depth`: the perpendicular distance from the fissure sulcus bottom to the opening line.

Numeric fields are written to `batch_inference_summary.csv` as
`fissure_width_px`, `fissure_depth_px`, `fissure_mean_width_px`, and mm fields
when pixel spacing is provided.

Measure saved masks directly:

```bash
python scripts/measure_output_masks.py ^
  --input-root "Results\data_260513" ^
  --output-dir "Results\data_260513\measurement_by_metric" ^
  --overwrite
```

For multiclass masks, the direct script uses class `1` for lateral fissure
metrics and class `2` for longitudinal fissure metrics by default. Override
them with `--lateral-class` and `--longitudinal-class` if the label definition
changes.
