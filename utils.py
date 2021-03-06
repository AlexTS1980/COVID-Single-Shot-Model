import torch
import torch.nn

########################   AVERAGE PRECISION COMPUTATION ########################
# adapted from Matterport Mask R-CNN implementation                             #
# https://github.com/matterport/Mask_RCNN                                       #
# inputs are predicted masks>threshold (0.5)                                    #
#################################################################################
def compute_overlaps_masks(masks1, masks2):
    # masks1: (HxWxnum_pred)
    # masks2: (HxWxnum_gts)
    # flatten masks and compute their areas
    # masks1: num_pred x H*W
    # masks2: num_gt x H*W
    # overlap: num_pred x num_gt
    masks1 = masks1.flatten(start_dim=1)
    masks2 = masks2.flatten(start_dim=1)
    area2 = masks2.sum(dim=(1,), dtype=torch.float)
    area1 = masks1.sum(dim=(1,), dtype=torch.float)
    # duplicatae each predicted mask num_gt times, compute the union (sum) of areas
    # num_pred x num_gt
    area1 = area1.unsqueeze_(1).expand(*[area1.size()[0], area2.size()[0]])
    union = area1 + area2
    # intersections and union: transpose predictions, the overlap matrix is num_predxnum_gts
    intersections = masks1.float().matmul(masks2.t().float())
    # +1: divide by 0
    overlaps = intersections / (union-intersections)
    return overlaps


# compute average precision for the  specified IoU threshold
def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5):
    # Sort predictions by score from high to low
    indices = pred_scores.argsort().flip(dims=(0,))
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[indices,...]
    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    # separate predictions for each gt object (a total of gt_masks splits
    split_overlaps = overlaps.t().split(1)
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    # At the start all predictions are False Positives, all gts are False Negatives
    pred_match = torch.tensor([-1]).expand(pred_boxes.size()[0]).float()
    gt_match = torch.tensor([-1]).expand(gt_boxes.size()[0]).float()
    # Alex: loop through each column (gt object), get
    for _i, splits in enumerate(split_overlaps):
        # ground truth class
        gt_class = gt_class_ids[_i]
        if (splits>iou_threshold).any():
           # get best predictions, their indices inthe IoU tensor and their classes
           global_best_preds_inds = torch.nonzero(splits[0]>iou_threshold).view(-1)
           pred_classes = pred_class_ids[global_best_preds_inds]
           best_preds = splits[0][splits[0]>iou_threshold]
           #  sort them locally-nothing else,
           local_best_preds_sorted = best_preds.argsort().flip(dims=(0,))
           # loop through each prediction's index, sorted in the descending order
           for p in local_best_preds_sorted:
               if pred_classes[p]==gt_class:
                  # Hit?
                  match_count +=1
                  pred_match[global_best_preds_inds[p]] = _i
                  gt_match[_i] = global_best_preds_inds[p]
                  # important: if the prediction is True Positive, finish the loop
                  break

    return gt_match, pred_match, overlaps


# AP for a single IoU threshold and 1 image
def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):

    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = (pred_match>-1).cumsum(dim=0).float().div(torch.arange(pred_match.numel()).float()+1)
    recalls = (pred_match>-1).cumsum(dim=0).float().div(gt_match.numel())
    # Pad with start and end values to simplify the math
    precisions = torch.cat([torch.tensor([0]).float(), precisions, torch.tensor([0]).float()])
    recalls = torch.cat([torch.tensor([0]).float(), recalls, torch.tensor([1]).float()])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    # Compute mean AP over recall range
    indices = torch.nonzero(recalls[:-1] !=recalls[1:]).squeeze_(1)+1
    mAP = torch.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    return mAP, precisions, recalls, overlaps

# easier boolean argument
def str2bool(v):
    return v.lower() in ('true')

# try copying weights from the segmentation model during training
def copy_weights(joint_lw_model):
    # from
    segmentation_layers=[joint_lw_model.roi_heads.box_head.fc6.weight, joint_lw_model.roi_heads.box_head.fc6.bias,
        joint_lw_model.roi_heads.box_head.fc7.weight, joint_lw_model.roi_heads.box_head.fc7.bias]
    segmentation_pred_layers = [joint_lw_model.roi_heads.box_predictor.cls_score.weight, joint_lw_model.roi_heads.box_predictor.cls_score.bias,
        joint_lw_model.roi_heads.box_predictor.bbox_pred.weight, joint_lw_model.roi_heads.box_predictor.bbox_pred.bias]
    # to
    classification_layers=[joint_lw_model.roi_heads.box_head_s2.fc6.weight, joint_lw_model.roi_heads.box_head_s2.fc6.bias,
        joint_lw_model.roi_heads.box_head_s2.fc7.weight, joint_lw_model.roi_heads.box_head_s2.fc7.bias]
    pred_layers = [joint_lw_model.roi_heads.box_predictor_s2.cls_score.weight, joint_lw_model.roi_heads.box_predictor_s2.cls_score.bias,
        joint_lw_model.roi_heads.box_predictor_s2.bbox_pred.weight, joint_lw_model.roi_heads.box_predictor_s2.bbox_pred.bias]
    #
    # grads off => copy weights => grads on
    #
    for id,l in enumerate(classification_layers):
        l.requires_grad_(False)
        segmentation_layers[id].requires_grad_(False)
        l.copy_(segmentation_layers[id])
        l.requires_grad_(True)
        segmentation_layers[id].requires_grad_(True)

    for jd,m in enumerate(pred_layers):
        m.requires_grad_(False)
        segmentation_pred_layers[jd].requires_grad_(False)
        m.copy_(segmentation_pred_layers[jd])
        m.requires_grad_(True)
        segmentation_pred_layers[jd].requires_grad_(True)


