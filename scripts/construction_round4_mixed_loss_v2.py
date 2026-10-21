"""Two-class true-mask / publisher-box loss, without box-filled segmentation targets.

Author class order is immutable: crack=0, spalling=1. A box-only camera frame
does not establish the other class as absent. Its BCE channel is explicitly
unsupervised, including background anchors. This module changes no installed
Ultralytics sources and never constructs a model or opens data on import.
"""
import torch
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss


CLASSES = ('crack', 'spalling')


def balanced_true_mask_loss(logits, target):
    """Actual ROI pixels only: equal positive/negative mass BCE plus soft Dice.

    Thin positive pixels receive half the BCE weight even when a curved crack's
    bounding rectangle contains much background. This never alters mask pixels,
    dilates a stroke or fabricates a rectangular foreground target.
    """
    # OpenCV INTER_AREA can produce1+one float32 ULP from a binary source mask.
    # Reject genuine bad coverage, then correct ONLY bounded arithmetic noise.
    # Source NPZ/masks remain unchanged; no thresholding or stroke dilation.
    tolerance = 1e-6
    if target.shape != logits.shape or not torch.isfinite(target).all() or not torch.isfinite(logits).all() or torch.any(target < -tolerance) or torch.any(target > 1 + tolerance):
        raise ValueError('Actual fractional source mask must match finite ROI logits')
    target = target.clamp(0, 1)
    positive_mass = target.sum()
    negative_mass = (1 - target).sum()
    if not positive_mass > 0:
        raise ValueError('Visible annotated mask has no positive source coverage inside its box')
    positive = (nn.functional.softplus(-logits) * target).sum() / positive_mass
    negative = (nn.functional.softplus(logits) * (1 - target)).sum() / negative_mass.clamp_min(1)
    balanced_bce = (positive + negative) / 2 if negative_mass > 0 else positive
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + positive_mass + 1)
    return .5 * balanced_bce + .5 * dice


class PartialClassBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.supervised_classes = None
        self.positive_assignments_only = False

    def forward(self, logits, target):
        known = self.supervised_classes
        if known is None or known.shape != (logits.shape[0], logits.shape[-1]):
            raise ValueError('Explicit per-image class supervision is required')
        if not torch.all((known == 0) | (known == 1)):
            raise ValueError('Supervision mask must be binary')
        selected = known[:, None, :]
        if self.positive_assignments_only:
            selected = selected * (target > 0).to(logits.dtype)
        self.last_selected_anchors = selected.expand_as(logits).detach()
        return nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none') * selected


class MixedConstructionLoss(v8SegmentationLoss):
    """Batch1 supervision branches share the detector; masks are real source GT only."""
    def __init__(self, model):
        super().__init__(model)
        if self.nc != 2 or self.overlap:
            raise ValueError('Require exactly two author classes and independent instance masks')
        self.bce = PartialClassBCE()

    def get_assigned_targets_and_loss(self, preds, batch, **kwargs):
        result = super().get_assigned_targets_and_loss(preds, batch, **kwargs)
        foreground, gt_indices = result[0][:2]
        assigned = sorted(set(gt_indices[foreground].detach().cpu().tolist()))
        self.last_assignment = {'foreground_anchors': int(foreground.sum()), 'actual_gt_instances': len(batch['cls']),
            'assigned_gt_indices': assigned, 'unassigned_gt_indices': [i for i in range(len(batch['cls'])) if i not in assigned]}
        return result

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
        # Same parent assignment/normalization, but do not advanced-index-copy a
        # full640 GT plane for every positive anchor (potentially hundreds).
        # Each anchor reads a view of its one actual source-instance plane.
        total = proto.sum() * 0
        mask_h, mask_w = proto.shape[-2:]
        normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        coordinates = normalized * proto.new_tensor([mask_w, mask_h, mask_w, mask_h])
        for image_index in range(len(proto)):
            original_masks = masks if len(proto) == 1 else masks[batch_idx.view(-1) == image_index]
            for anchor in torch.where(fg_mask[image_index])[0]:
                gt_index = int(target_gt_idx[image_index, anchor])
                total = total + self.single_mask_loss(original_masks[gt_index:gt_index + 1],
                    pred_masks[image_index, anchor:anchor + 1], proto[image_index],
                    coordinates[image_index, anchor:anchor + 1], None)
        return total / fg_mask.sum()

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        # Parent criterion upsamples the actual32 prototypes to the640 source-GT
        # plane. Compute only each actual GT box ROI to avoid allocating all
        # foreground-anchor × full640 maps. GT never changes network input.
        total = proto.sum() * 0
        height, width = proto.shape[-2:]
        for index in range(len(pred)):
            x1, y1, x2, y2 = xyxy[index]
            left = max(0, int(torch.floor(x1).detach()))
            top = max(0, int(torch.floor(y1).detach()))
            right = min(width, int(torch.ceil(x2).detach()))
            bottom = min(height, int(torch.ceil(y2).detach()))
            if right <= left or bottom <= top:
                raise ValueError('Actual matched source mask ROI has no positive geometry')
            roi_proto = proto[:, top:bottom, left:right]
            logits = torch.einsum('c,chw->hw', pred[index], roi_proto)
            target = gt_mask[index, top:bottom, left:right]
            total = total + balanced_true_mask_loss(logits, target)
        return total

    def loss(self, preds, batch):
        if preds['boxes'].shape[0] != 1:
            raise ValueError('This version prerequires batch1 homogeneous supervision')
        mode = batch.get('supervision_kind')
        if mode not in {'actual_masks', 'publisher_boxes', 'verified_target_negative'}:
            raise ValueError('Unknown source supervision kind')
        known = batch['supervised_classes'].to(device=self.device, dtype=preds['scores'].dtype)
        if known.shape != (1, 2) or not known.any():
            raise ValueError('At least one explicitly annotated class is required')
        raw_classes = batch['cls'].flatten()
        if not torch.isfinite(raw_classes).all() or not torch.all(raw_classes == raw_classes.round()):
            raise ValueError('GT class IDs must be finite exact integers before conversion')
        classes = raw_classes.long()
        if len(classes) and (not torch.all((classes >= 0) & (classes < 2)) or not torch.all(known[0, classes] == 1)):
            raise ValueError('GT class falls outside explicit source annotation scope')
        self.bce.supervised_classes = known
        self.bce.positive_assignments_only = mode == 'publisher_boxes'
        if mode == 'actual_masks':
            if 'masks' not in batch or len(batch['masks']) != len(classes):
                raise ValueError('Every actual mask must match an annotated target')
            if batch.get('mask_origin') not in {'publisher_instance_polygon', 'publisher_semantic_polygon_region'}:
                raise ValueError('Actual source mask provenance is mandatory; box filling is forbidden')
            return super().loss(preds, batch)
        if 'masks' in batch:
            raise ValueError('Box-only / negative samples must not supply invented masks')
        if mode == 'verified_target_negative' and (len(classes) or not torch.all(known == 1)):
            raise ValueError('Verified two-class negative must have complete annotation and no target')
        detection, _ = v8DetectionLoss.loss(self, preds, batch)
        proto = preds['proto']
        if isinstance(proto, tuple):
            raise ValueError('Unexpected semantic head; source YOLOv8n-seg has no semantic output')
        zero_mask = (proto * 0).sum() + (preds['mask_coefficient'] * 0).sum()
        combined = torch.stack((detection[0], zero_mask, detection[1], detection[2], zero_mask))
        return combined, dict(zip(self.loss_names, combined.detach()))
