import unittest

import torch

from utils.entropy import normalized_entropy_map


class NormalizedEntropyMapTest(unittest.TestCase):
    def test_three_class_one_hot_entropy_is_zero(self):
        p = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
        self.assertAlmostEqual(float(normalized_entropy_map(p).item()), 0.0, places=6)

    def test_three_class_uniform_entropy_is_one(self):
        p = torch.full((1, 3, 1, 1), 1.0 / 3.0)
        self.assertAlmostEqual(float(normalized_entropy_map(p).item()), 1.0, places=6)

    def test_binary_uniform_entropy_is_one(self):
        p = torch.full((1, 2, 1, 1), 0.5)
        self.assertAlmostEqual(float(normalized_entropy_map(p).item()), 1.0, places=6)

    def test_random_softmax_output_is_bounded(self):
        p = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
        out = normalized_entropy_map(p)
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_backward_has_finite_gradient(self):
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        p = torch.softmax(logits, dim=1)
        normalized_entropy_map(p).mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            normalized_entropy_map(torch.ones(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
