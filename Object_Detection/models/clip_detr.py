"""
CLIP-DETR model: A complete object detection model using CLIP features and Q-Former architecture.
Includes the model, criterion, and post-processing.

This version removes unused auxiliary layers and simplifies the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_projector import ClipProjector
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from util import box_ops
from util.misc import accuracy, is_dist_avail_and_initialized, get_world_size
from .qformer import QFormer


class CLIPDETR(nn.Module):
    """
    Object detection model combining CLIP features with Q-Former and DETR-like detection heads.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        num_classes: int = 91,
        num_queries: int = 100,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
    ):
        super().__init__()
        
        # CLIP visual encoder with projection
        # output: projected_features[batch_size, num_tokens, projection_dim], mask[batch_size, num_tokens]
        self.clip_projector = ClipProjector(
            clip_model_name=clip_model_name,
            projection_dim=hidden_dim,
            #layer=-1  # Use last layer features
        )
    
        # Build Q-Former (single transformer decoder layer with learnable queries)
        self.qformer = QFormer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=pre_norm,
        )
        
        # Learnable query embeddings
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.class_projection = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                        nn.ReLU(),                                   
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),                                   
                                        nn.Linear(hidden_dim, 4)
                                        )

        
    def forward(self, samples: torch.Tensor):
        """
        Forward pass for object detection.
        
        Args:
            samples: Batched images, of shape [batch_size, 3, H, W]
            
        Returns:
            Dict containing:
                - "pred_logits": classification logits [batch_size x num_queries x num_classes]
                - "pred_boxes": normalized boxes [batch_size x num_queries x 4] in (cx, cy, w, h) format
        """
        # Extract features from CLIP
        patch_features, token_mask = self.clip_projector(samples) # output: [batch_size, num_tokens, projection_dim], [batch_size, num_tokens]
        
        # No positional encoding is used since we rely on CLIP's internal positional encoding
        pos_embed = None
        
        # Forward through the Q-Former
        # The QFormer uses the CLIP tokens as memory for cross-attention
        # ouitput: output [B, num_queries, hidden_dim],  src [B, num_tokens, hidden_dim]
        hs, memory = self.qformer(patch_features, token_mask, self.query_embed, pos_embed) 
        
        # Predict class labels and bounding boxes
        outputs_class = self.class_projection(hs) # output: [B, num_queries, num_class]
        outputs_coord = self.bbox_projection(hs).sigmoid()  # Sigmoid to bound between [0, 1], # output: [B, num_queries, 4]
        
        # Format output
        out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
        
        return out



class SetCriterion(nn.Module):
    """
    Loss computation for DETR-style object detection.
    Performs Hungarian matching between predicted and ground-truth objects,
    then computes classification and box losses.
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Initialize the criterion.
        
        Args:
            num_classes: Number of object categories
            matcher: Module to compute matching between targets and predictions
            weight_dict: Dict containing weights for different losses
            eos_coef: Weight for no-object category
            losses: List of losses to apply
            
            weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
            weight_dict['loss_giou'] = args.giou_loss_coef
            losses = ['labels', 'boxes', 'cardinality']
    
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Prepare weight for classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # Set weight for no-object class
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # Extract indices of matched pairs
        idx = self._get_src_permutation_idx(indices)
        
        # Get target classes for matched pairs
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Create target tensor with shape [batch_size, num_queries]
        # Initialize all to no-object class (num_classes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        
        # Set matched pairs to their actual classes
        target_classes[idx] = target_classes_o
        
        # Compute cross-entropy loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}
        
        # Optional logging of class error
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            
        return losses
        
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute bounding box losses: L1 and GIoU
        """
        assert 'pred_boxes' in outputs
        
        # Extract indices of matched pairs
        idx = self._get_src_permutation_idx(indices)
        
        # Get predicted boxes for matched pairs
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target boxes for matched pairs
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # Compute GIoU loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
        
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute cardinality error (number of predicted objects vs ground truth)
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        # Count number of ground truth objects
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        # Count number of predictions that are not no-object
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        
        # L1 loss between predicted counts and ground truth counts
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        
        return losses
        
    def _get_src_permutation_idx(self, indices):
        """
        Extract source indices from matched pairs.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
    def _get_tgt_permutation_idx(self, indices):
        """
        Extract target indices from matched pairs.
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Compute specified loss.
        """
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        
    def forward(self, outputs, targets):
        """
        Forward pass to compute all losses.
        
        Args:
            outputs: Dict of model outputs
            targets: List of target dicts
            
        Returns:
            Dict of losses
        """
        # Remove auxiliary outputs for matching (not needed with single layer)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Find matching between predictions and targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute average number of target boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Handle distributed training
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            
        # We've removed the aux_outputs processing since we're using a single layer
        
        return losses


class PostProcess(nn.Module):
    """
    Post-processing module to convert model outputs to a format expected by COCO API.
    Converts normalized box coordinates to pixel coordinates.
    """
    def __init__(self, image_input_size):
        super().__init__()
        self.image_input_size = image_input_size
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Post-process outputs to get final detections.

        Args:
            outputs: Model outputs
            target_sizes: Tensor of size [batch_size, 2] with image sizes (h, w)

        Returns:
            List of dicts with processed detections
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # Get probabilities
        prob = F.softmax(out_logits, -1)

        # Get scores and labels (excluding no-object class)
        scores, labels = prob[..., :-1].max(-1)

        # Convert center-size format to x1,y1,x2,y2 format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # 각 이미지에 대해 처리
        results = []
        for i, (box, target_size) in enumerate(zip(boxes, target_sizes)):
            img_h, img_w = target_size

            # 패딩 정보 계산
            max_size = max(img_h.item(), img_w.item())
            left_pad = (max_size - img_w.item()) // 2
            top_pad = (max_size - img_h.item()) // 2

            # 0-1 정규화된 좌표를 픽셀 좌표로 변환 (336x336)
            box_scaled = box * self.image_input_size

            # 리사이징 비율 계산
            scale = self.image_input_size / max_size

            # 336 -> 원본 정사각형 크기로 역스케일링
            box_descaled = box_scaled / scale

            # 패딩 제거
            adjusted_box = box_descaled.clone()
            adjusted_box[:, 0] = torch.clamp(box_descaled[:, 0] - left_pad, min=0)  # x1
            adjusted_box[:, 2] = torch.clamp(box_descaled[:, 2] - left_pad, max=img_w)  # x2
            adjusted_box[:, 1] = torch.clamp(box_descaled[:, 1] - top_pad, min=0)   # y1
            adjusted_box[:, 3] = torch.clamp(box_descaled[:, 3] - top_pad, max=img_h)   # y2

            results.append({
                'scores': scores[i],
                'labels': labels[i],
                'boxes': adjusted_box
            })

        return results

def build_model(args):
    """
    Build the CLIP-DETR model and associated components.
    
    Args:
        args: Arguments containing model configuration
        
    Returns:
        model: The CLIP-DETR model
        criterion: Loss criterion for training
        postprocessors: Post-processing modules for evaluation
    """
    from .matcher import build_matcher
    
    # Determine number of classes based on dataset
    num_classes = 91 if args.dataset_file == 'coco' else args.num_classes
    
    # Create model
    model = CLIPDETR(
        clip_model_name=args.clip_model_name,
        num_classes=num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
    ) 
    # input: [batch_size, 3, H, W]
    # output: {
    #        'pred_logits': outputs_class, => logit
    #        'pred_boxes': outputs_coord => sigmoid(0-1)
    #    }
        
    
    # Build matcher for Hungarian matching
    matcher = build_matcher(args)
    
    # Define weight dictionary for different losses
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # Define losses
    losses = ['labels', 'boxes', 'cardinality']
    
    # Create criterion
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )
    
    # Define postprocessors for evaluation
    postprocessors = {'bbox': PostProcess(args.image_input_size)}
    
    return model, criterion, postprocessors