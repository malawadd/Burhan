"""Exact same real-mask objective, batching positive anchors sharing an actual GT ROI."""
import torch
from torch import nn
from construction_round4_mixed_loss_v2 import (
    PartialClassBCE, balanced_true_mask_loss, MixedConstructionLoss as ScalarMixedLoss,
)


def grouped_true_mask_loss(logits, target):
    """Sum the unchanged v2 objective independently over K matched-anchor masks."""
    tolerance = 1e-6
    if logits.ndim != 3 or target.shape != logits.shape[-2:] or not torch.isfinite(target).all() or not torch.isfinite(logits).all() or torch.any(target < -tolerance) or torch.any(target > 1 + tolerance):
        raise ValueError('Actual fractional source mask must match finite grouped ROI logits')
    target = target.clamp(0, 1)
    positive_mass = target.sum()
    negative_mass = (1 - target).sum()
    if not positive_mass > 0:
        raise ValueError('Visible annotated mask has no positive source coverage inside its box')
    dimensions = (-2, -1)
    positive = (nn.functional.softplus(-logits) * target).sum(dimensions) / positive_mass
    negative = (nn.functional.softplus(logits) * (1 - target)).sum(dimensions) / negative_mass.clamp_min(1)
    balanced = (positive + negative) / 2 if negative_mass > 0 else positive
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum(dimensions) + 1) / (probability.sum(dimensions) + positive_mass + 1)
    return (.5 * balanced + .5 * dice).sum()


class MixedConstructionLoss(ScalarMixedLoss):
    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
        # Once per image, rather than a full32×640×640 zero-gradient path and
        # slice-backward scatter for every one of potentially hundreds of anchors.
        total = proto.sum() * 0
        height, width = proto.shape[-2:]
        coordinates = target_bboxes / imgsz[[1, 0, 1, 0]] * proto.new_tensor([width, height, width, height])
        for image_index in range(len(proto)):
            actual_masks = masks if len(proto) == 1 else masks[batch_idx.view(-1) == image_index]
            anchors = torch.where(fg_mask[image_index])[0]
            assigned = target_gt_idx[image_index, anchors]
            for gt_index in torch.unique(assigned):
                selected = anchors[assigned == gt_index]
                boxes = coordinates[image_index, selected]
                if not torch.all(boxes == boxes[0]):
                    raise ValueError('Same GT index unexpectedly has different true ROI coordinates')
                x1, y1, x2, y2 = boxes[0]
                left, top = max(0, int(torch.floor(x1).detach())), max(0, int(torch.floor(y1).detach()))
                right, bottom = min(width, int(torch.ceil(x2).detach())), min(height, int(torch.ceil(y2).detach()))
                if right <= left or bottom <= top:
                    raise ValueError('Actual matched GT ROI has no positive geometry')
                coefficients = pred_masks[image_index, selected]
                features = proto[image_index, :, top:bottom, left:right]
                logits = torch.einsum('kc,chw->khw', coefficients, features)
                target = actual_masks[int(gt_index), top:bottom, left:right]
                total = total + grouped_true_mask_loss(logits, target)
        return total / fg_mask.sum()
