import torch
import torch.nn as nn
from torch.nn import functional as F


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                     F.softmax(out_t / self.T, dim=1), reduction="batchmean")
            * self.T
            * self.T
        )
        return loss


def loss_diff1(u_prediction_1, u_prediction_2):
    target = u_prediction_2.detach().float()
    prediction = u_prediction_1.clamp(1e-8, 1 - 1e-7)
    return F.binary_cross_entropy(prediction, target, reduction="mean")


def loss_diff2(u_prediction_1, u_prediction_2):
    target = u_prediction_2.detach().float()
    prediction = u_prediction_1.clamp(1e-8, 1 - 1e-7)
    return F.binary_cross_entropy(prediction, target, reduction="mean")


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, target):
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.ndim != 3:
            raise ValueError(f"DiceLoss target must have shape [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
        one_hot = F.one_hot(target.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        return one_hot.float()

    def _expand_mask(self, mask, reference):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"DiceLoss mask must have shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
        return mask.float().expand_as(reference)

    def _dice_loss(self, score, target, mask=None):
        target = target.float()
        smooth = 1e-10
        if mask is not None:
            mask = mask.float()
            score = score * mask
            target = target * mask
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: {tuple(inputs.size())} vs {tuple(target.size())}")
        expanded_mask = self._expand_mask(mask, target) if mask is not None else None
        loss = 0.0
        for i in range(0, self.n_classes):
            class_mask = expanded_mask[:, i] if expanded_mask is not None else None
            dice = self._dice_loss(inputs[:, i], target[:, i], class_mask)
            loss += dice * weight[i]
        return loss / self.n_classes


def dice_loss(pred, label, epsilon=1e-5):
    intersection = torch.sum(pred * label, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(label, dim=(2, 3))
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss_value = 1.0 - dice_coefficient
    return dice_loss_value.mean()


def sigmoid_mse_loss_map(input_logits, target_logits):
    if input_logits.size() != target_logits.size():
        raise ValueError(f"input_logits and target_logits must match, got {input_logits.size()} and {target_logits.size()}")
    input_softmax = torch.nn.Sigmoid()(input_logits)
    target_softmax = torch.nn.Sigmoid()(target_logits)
    mse_loss_map = (input_softmax - target_softmax) ** 2
    return mse_loss_map


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)
