"""Training-only SSDLite loss that learns from zero-object photographs.

Adapted from torchvision.models.detection.ssd.SSD.compute_loss (BSD-3-Clause).
Unlike the original loss, empty targets contribute three hard-background anchors.
Inference, anchors, matching, scores and NMS are unchanged.

BSD 3-Clause License
Copyright (c) Soumith Chintala 2016. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch.nn import functional as F


def background_aware_loss(self, targets, head_outputs, anchors, matched_idxs):
    regression = head_outputs["bbox_regression"]
    logits = head_outputs["cls_logits"]
    foreground_count = 0
    regression_losses, classification_targets = [], []
    for target, offsets, scores, anchor, matches in zip(targets, regression, logits, anchors, matched_idxs):
        foreground = torch.where(matches >= 0)[0]
        matched = matches[foreground]
        foreground_count += matched.numel()
        expected_offsets = self.box_coder.encode_single(target["boxes"][matched], anchor[foreground])
        regression_losses.append(F.smooth_l1_loss(offsets[foreground], expected_offsets, reduction="sum"))
        labels = torch.zeros(scores.shape[0], dtype=target["labels"].dtype, device=scores.device)
        labels[foreground] = target["labels"][matched]
        classification_targets.append(labels)
    labels = torch.stack(classification_targets)
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
    foreground = labels > 0
    # The stock SSD expression yields zero background samples on an empty image.
    requested = (self.neg_to_pos_ratio * foreground.sum(1, keepdim=True)).clamp(min=3)
    available = (~foreground).sum(1, keepdim=True)
    requested = torch.minimum(requested, available)
    negative_losses = losses.detach().clone()
    negative_losses[foreground] = -float("inf")
    order = negative_losses.argsort(1, descending=True).argsort(1)
    background = (order < requested) & ~foreground
    # An all-negative batch must not be amplified by a one-anchor denominator.
    denominator = max(len(targets), foreground_count)
    self.background_loss_diagnostic = {
        "foreground_anchors": foreground_count,
        "background_anchors_by_image": background.sum(1).tolist(),
        "denominator": denominator,
    }
    return {"bbox_regression": torch.stack(regression_losses).sum() / denominator,
            "classification": (losses[foreground].sum() + losses[background].sum()) / denominator}


def install(model):
    """Change only training loss on this instance; do not patch site-packages."""
    model.compute_loss = MethodType(background_aware_loss, model)
    return model
