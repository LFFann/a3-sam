import unittest

import torch

from utils.losses import DiceLoss


class DiceLossTest(unittest.TestCase):
    def test_perfect_prediction_has_near_zero_loss(self):
        target = torch.tensor([[[0, 1], [2, 1]]])
        inputs = torch.nn.functional.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
        loss = DiceLoss(3)(inputs, target)
        self.assertLess(float(loss.item()), 1e-6)

    def test_non_overlapping_prediction_has_large_loss(self):
        target = torch.tensor([[[1, 1], [1, 1]]])
        pred_labels = torch.zeros_like(target)
        inputs = torch.nn.functional.one_hot(pred_labels, num_classes=3).permute(0, 3, 1, 2).float()
        loss = DiceLoss(3)(inputs, target)
        self.assertGreater(float(loss.item()), 0.3)

    def test_one_hot_shape_is_correct_for_three_classes(self):
        target = torch.tensor([[[0, 1], [2, 0]]])
        one_hot = DiceLoss(3)._one_hot_encoder(target)
        self.assertEqual(tuple(one_hot.shape), (1, 3, 2, 2))

    def test_background_class_zero_is_encoded_correctly(self):
        target = torch.tensor([[[0, 1], [2, 0]]])
        one_hot = DiceLoss(3)._one_hot_encoder(target)
        self.assertEqual(int(one_hot[:, 0].sum().item()), 2)

    def test_masked_pixel_does_not_contribute(self):
        target = torch.tensor([[[1, 1]]])
        pred_labels = torch.tensor([[[1, 0]]])
        inputs = torch.nn.functional.one_hot(pred_labels, num_classes=2).permute(0, 3, 1, 2).float()
        mask = torch.tensor([[[1, 0]]])
        loss = DiceLoss(2)(inputs, target, mask=mask)
        self.assertLess(float(loss.item()), 1e-6)

    def test_backward_has_finite_gradient(self):
        target = torch.tensor([[[0, 1], [2, 1]]])
        logits = torch.randn(1, 3, 2, 2, requires_grad=True)
        loss = DiceLoss(3)(torch.softmax(logits, dim=1), target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
