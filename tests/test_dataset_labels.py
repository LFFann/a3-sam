import unittest

import numpy as np

from utils.label_validation import validate_class_index_mask


class DatasetLabelValidationTest(unittest.TestCase):
    def test_valid_class_indices_pass(self):
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        out = validate_class_index_mask(mask, 3, source="valid")
        self.assertTrue(np.array_equal(out, mask.astype(np.int64)))

    def test_legacy_uint8_encoding_raises(self):
        with self.assertRaises(ValueError):
            validate_class_index_mask(np.array([[0, 127, 255]], dtype=np.uint8), 3, source="legacy")

    def test_negative_label_raises(self):
        with self.assertRaises(ValueError):
            validate_class_index_mask(np.array([[0, -1]], dtype=np.int64), 3, source="negative")

    def test_label_equal_num_classes_raises(self):
        with self.assertRaises(ValueError):
            validate_class_index_mask(np.array([[0, 3]], dtype=np.int64), 3, source="too_large")

    def test_integer_float_values_are_allowed(self):
        mask = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        out = validate_class_index_mask(mask, 3, source="float_int")
        self.assertEqual(out.dtype, np.int64)
        self.assertTrue(np.array_equal(out, np.array([[0, 1, 2]], dtype=np.int64)))

    def test_fractional_float_raises(self):
        with self.assertRaises(ValueError):
            validate_class_index_mask(np.array([[0.0, 1.2]], dtype=np.float32), 3, source="fractional")

    def test_rgb_mask_raises_with_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_class_index_mask(np.zeros((2, 2, 3), dtype=np.uint8), 3, source="rgb")


if __name__ == "__main__":
    unittest.main()
