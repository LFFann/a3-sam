import math
import unittest

import numpy as np

from prediction_multiclass import evaluate_multiclass
from utils.utils import _binary_class_metrics, multiclass_segmentation_metrics, safe_binary_hd95


class MetricsTest(unittest.TestCase):
    def test_identical_nonempty_mask_hd95_is_zero(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        self.assertAlmostEqual(safe_binary_hd95(mask, mask), 0.0, places=6)

    def test_shifted_square_hd95_is_one_pixel(self):
        gt = np.zeros((8, 8), dtype=np.uint8)
        pred = np.zeros((8, 8), dtype=np.uint8)
        gt[2:5, 2:5] = 1
        pred[2:5, 3:6] = 1
        self.assertAlmostEqual(safe_binary_hd95(pred, gt), 1.0, places=6)

    def test_both_empty_hd95_is_zero(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(safe_binary_hd95(mask, mask), 0.0)

    def test_only_pred_empty_hd95_is_nan(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[1, 1] = 1
        self.assertTrue(math.isnan(safe_binary_hd95(pred, gt)))

    def test_only_gt_empty_hd95_is_nan(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred[1, 1] = 1
        self.assertTrue(math.isnan(safe_binary_hd95(pred, gt)))

    def test_prediction_and_training_metrics_match(self):
        gt = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 2]], dtype=np.uint8)
        pred = np.array([[0, 1, 0], [0, 2, 2], [0, 0, 1]], dtype=np.uint8)
        pred_metrics = evaluate_multiclass(gt, pred, 3)
        train_metrics = multiclass_segmentation_metrics(gt, pred, 3)
        self.assertAlmostEqual(pred_metrics["dice"], train_metrics["avg_dice"], places=6)
        self.assertAlmostEqual(pred_metrics["iou"], train_metrics["avg_iou"], places=6)
        self.assertAlmostEqual(pred_metrics["hd95"], train_metrics["avg_hd95"], places=6)

    def test_dice_and_iou_behavior_unchanged(self):
        gt = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        dice, iou, _ = _binary_class_metrics(gt == 1, pred == 1)
        self.assertAlmostEqual(dice, 0.8, places=6)
        self.assertAlmostEqual(iou, 2.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
