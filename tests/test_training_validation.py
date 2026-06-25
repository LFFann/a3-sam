import unittest

from utils.training_validation import validate_ugda_batch_config, validate_ugda_runtime_batch


class TrainingValidationTest(unittest.TestCase):
    def test_valid_configs_pass(self):
        validate_ugda_batch_config(32, 16)
        validate_ugda_batch_config(16, 8)

    def test_unequal_halves_raise(self):
        with self.assertRaises(ValueError):
            validate_ugda_batch_config(40, 16)

    def test_no_unlabeled_half_raises(self):
        with self.assertRaises(ValueError):
            validate_ugda_batch_config(16, 16)

    def test_zero_batch_raises(self):
        with self.assertRaises(ValueError):
            validate_ugda_batch_config(0, 16)

    def test_runtime_mismatch_raises_before_reshape(self):
        with self.assertRaisesRegex(ValueError, "actual_batch_size=40"):
            validate_ugda_runtime_batch(40, 16)


if __name__ == "__main__":
    unittest.main()
