# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
This module is essential and doesn't need significant modifications.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from center-size format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        x: Tensor of shape (..., 4) containing bboxes in center-size format
        
    Returns:
        Tensor of same shape containing bboxes in corner format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from corner format (x1, y1, x2, y2) to center-size format (cx, cy, w, h).
    
    Args:
        x: Tensor of shape (..., 4) containing bboxes in corner format
        
    Returns:
        Tensor of same shape containing bboxes in center-size format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
        
    Returns:
        Tuple of:
            - IoU: Tensor of shape (N, M) with pairwise IoUs
            - Union: Tensor of shape (N, M) with pairwise unions
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
        
    Returns:
        Tensor of shape (N, M) with pairwise GIoUs
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# def masks_to_boxes(masks):
#     """
#     Compute the bounding boxes around the provided masks.
    
#     Args:
#         masks: Tensor of shape (N, H, W) with binary masks
        
#     Returns:
#         Tensor of shape (N, 4) with bboxes in xyxy format
#     """
#     if masks.numel() == 0:
#         return torch.zeros((0, 4), device=masks.device)

#     h, w = masks.shape[-2:]

#     y = torch.arange(0, h, dtype=torch.float)
#     x = torch.arange(0, w, dtype=torch.float)
#     y, x = torch.meshgrid(y, x)

#     x_mask = (masks * x.unsqueeze(0))
#     x_max = x_mask.flatten(1).max(-1)[0]
#     x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

#     y_mask = (masks * y.unsqueeze(0))
#     y_max = y_mask.flatten(1).max(-1)[0]
#     y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

#     return torch.stack([x_min, y_min, x_max, y_max], 1)
