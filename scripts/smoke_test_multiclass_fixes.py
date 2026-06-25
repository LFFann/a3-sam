import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Model.discriminator import Discriminator
from utils.entropy import normalized_entropy_map
from utils.label_validation import validate_class_index_mask
from utils.training_validation import validate_ugda_batch_config
from utils.utils import _binary_class_metrics


def main() -> None:
    probabilities = torch.softmax(torch.randn(1, 3, 8, 8), dim=1)
    entropy = normalized_entropy_map(probabilities)
    assert entropy.shape == (1, 1, 8, 8)
    assert torch.all(entropy >= 0.0) and torch.all(entropy <= 1.0)

    discriminator = Discriminator(in_channels=3, out_conv_channels=3)
    discriminator.eval()
    pred_u = torch.randn(1, 3, 8, 8)
    pred_v = torch.randn(1, 3, 8, 8)
    soft_u = torch.softmax(pred_u, dim=1)
    soft_v = torch.softmax(pred_v, dim=1)
    ent_u = normalized_entropy_map(soft_u)
    ent_v = normalized_entropy_map(soft_v)
    fused = discriminator(pred_u, pred_v, soft_u, soft_v, ent_u, ent_v)
    assert fused.shape == (1, 3, 8, 8)

    gt = np.zeros((8, 8), dtype=np.uint8)
    pred = np.zeros((8, 8), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    pred[2:5, 3:6] = 1
    dice, iou, hd95 = _binary_class_metrics(gt == 1, pred == 1)
    assert 0.0 < dice < 1.0
    assert 0.0 < iou < 1.0
    assert math.isfinite(hd95)

    validate_ugda_batch_config(32, 16)
    try:
        validate_ugda_batch_config(40, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("UGDA 40/16 configuration should fail")

    try:
        validate_class_index_mask(np.array([[0, 127, 255]], dtype=np.uint8), 3, source="smoke")
    except ValueError:
        pass
    else:
        raise AssertionError("0/127/255 mask should fail class-index validation")

    print("Multiclass KnowSAM fix smoke test passed.")


if __name__ == "__main__":
    main()
