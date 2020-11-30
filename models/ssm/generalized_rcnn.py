# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
from .rpn import AnchorGenerator

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform,  s2new):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.s2new = s2new

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        if self.training:
           # targets for the segmenation and classification models
           if len(targets[0]['boxes']):
               images, targets = self.transform(images, targets=targets)
           # targets for the classification model only
           else:
               images, _ = self.transform(images, targets=None)
        # evaluation mode 
        else:
            images, _ = self.transform(images, targets=None)
        features = self.backbone(images.tensors)
        # Alex: LW Mask R-CNN outputs Cx16x16
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        # without the targets, RPN/RoI losses will be empty
        proposals, proposal_losses,_ = self.rpn(images, features, targets)
        detections, detector_losses, s2batch = self.roi_heads(features, proposals, images.image_sizes, targets)
        # Alex
        # Classification module
        if not self.training:
           scores_covid_boxes = self.s2new(s2batch[0]['ranked_boxes'])
           scores_covid_img = [dict(final_scores=scores_covid_boxes.squeeze_(0))]
        else:
           scores_covid_img=None
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, scores_covid_img, detections
