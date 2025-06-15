"""
CLIP-DETR model: A complete object detection model using CLIP features and Q-Former architecture.
Includes the model, criterion, and post-processing.

This version uses dynamic queries based on bounding boxes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_projector import ClipProjector
from .qformer import QFormer
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.keypoint_loss import OKSLoss
from util.misc import accuracy, is_dist_avail_and_initialized, get_world_size
import math

class BBoxPositionalEncoding(nn.Module):
    """
    Generate sinusoidal positional embeddings for bounding box coordinates.
    
    Each coordinate (cx, cy, w, h) is encoded to a d/2 dimension vector.
    Then all encodings are concatenated to a 2d dimension vector.
    Finally, an MLP projects it to a d dimension embedding.
    """
    def __init__(self, d_model, temp=10000):
        super().__init__()
        self.d_model = d_model
        self.temp = temp
        # 각 좌표당 d/2 차원의 인코딩 생성
        self.dim_per_coord = d_model // 2
        
        # MLP: 2d -> d -> d
        # 4개 좌표 * d/2 차원 = 2d 차원의 입력
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.dim_per_coord, d_model),  # 2d -> d
            nn.ReLU(),
            nn.Linear(d_model, d_model)  # d -> d
        )
        
    def _positional_encoding(self, x, dim):
        """
        단일 좌표에 대한 positional encoding 생성
        
        Args:
            x: Float tensor of shape [..., 1]
            dim: 생성할 인코딩의 차원
            
        Returns:
            encoding: Tensor of shape [..., dim]
        """
        assert dim % 2 == 0, "Dimension must be even"
        
        # Position encoding
        positions = x.unsqueeze(-1)  # [..., 1, 1]
        
        # 차원별 주파수 계산
        dim_t = torch.arange(0, dim, 2, dtype=torch.float, device=x.device)
        dim_t = self.temp ** (dim_t / dim)
        
        # 사인/코사인 인코딩 생성
        pos_enc = torch.zeros((*positions.shape[:-1], dim), device=x.device)
        pos_enc[..., 0::2] = torch.sin(positions / dim_t)  # 짝수 인덱스
        pos_enc[..., 1::2] = torch.cos(positions / dim_t)  # 홀수 인덱스
        
        return pos_enc.squeeze(-2)  # [..., dim]
    
    def forward(self, bbox):
        """
        바운딩 박스에 대한 positional encoding 생성
        
        Args:
            bbox: Tensor of shape [..., 4] - 정규화된 [cx, cy, w, h] 좌표
            
        Returns:
            encoding: Tensor of shape [..., d_model]
        """
        # 각 좌표 분리
        cx, cy, w, h = bbox[..., 0:1], bbox[..., 1:2], bbox[..., 2:3], bbox[..., 3:4]
        
        # 각 좌표에 대해 d/2 차원의 인코딩 생성
        pe_cx = self._positional_encoding(cx, self.dim_per_coord)  # [..., d/2]
        pe_cy = self._positional_encoding(cy, self.dim_per_coord)  # [..., d/2]
        pe_w = self._positional_encoding(w, self.dim_per_coord)    # [..., d/2]
        pe_h = self._positional_encoding(h, self.dim_per_coord)    # [..., d/2]
        
        # 모든 인코딩 연결 (총 2d 차원)
        concat_enc = torch.cat([pe_cx, pe_cy, pe_w, pe_h], dim=-1)  # [..., 2d]
        
        # MLP 통과시켜 최종 d 차원 인코딩 생성
        return self.mlp(concat_enc)  # [..., d]


class CLIPDETR(nn.Module):
    """
    Object detection model combining CLIP features with Q-Former and keypoint prediction heads.
    Using learnable query + sinusoidal bbox embeddings.
    Only predicts keypoints, not classes.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
        num_body_points: int = 17
    ):
        super().__init__()
        
        # CLIP visual encoder with projection
        self.clip_projector = ClipProjector(
            clip_model_name=clip_model_name, 
            projection_dim=hidden_dim,
        )
    
        # Build Q-Former
        self.qformer = QFormer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=pre_norm,
        )
        
        # Learnable query embedding (shared across all boxes)
        self.query_embed = nn.Embedding(1, hidden_dim)
        
        # MLP for positional encoding of bbox coordinates
        self.pe_layer = BBoxPositionalEncoding(hidden_dim)
        
        # Keypoint prediction heads
        self.keypoint_share_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.keypoint_split_projection = nn.ModuleList([
            nn.Linear(hidden_dim, 3)  # 각 키포인트별 개별 헤드 (x,y,visibility)
            for _ in range(num_body_points)
        ])
        
    def forward(self, samples: torch.Tensor, targets):
        """
        Forward pass using dynamic number of queries based on target bboxes.
        
        Args:
            samples: Batched images, of shape [batch_size, 3, H, W]
            targets: List of target dictionaries, each containing:
                    - 'boxes': [N, 4] bounding boxes in [cx, cy, w, h] format
                    - Other target information
            
        Returns:
            Dict containing:
                - 'pred_keypoints': [B, total_queries, num_bodypoints*3] in xyxyvv format
        """
        # Extract features from CLIP
        patch_features, _ = self.clip_projector(samples)  # [batch_size, num_tokens, hidden_dim], [batch_size, num_tokens]
        batch_size = samples.shape[0]
        
        # Generate queries based on target bboxes
        all_queries = []
        query_counts = []
        
        for i, target in enumerate(targets):
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]  # [N, 4], already in [cx, cy, w, h] format, normalized to 0-1
                num_boxes = boxes.shape[0]
                
                # Generate positional embeddings for bboxes
                bbox_embeddings = self.pe_layer(boxes)  # [N, hidden_dim]
                
                # Combine with learnable query (broadcast to match box count)
                learnable_query = self.query_embed.weight.expand(num_boxes, -1)  # [N, hidden_dim]
                queries = learnable_query + bbox_embeddings  # [N, hidden_dim]
                
                all_queries.append(queries)
                query_counts.append(num_boxes)
            else:
                # If no boxes, use zero queries for this image
                query_counts.append(0)
                all_queries.append(torch.zeros((0, self.query_embed.weight.shape[1]), device=samples.device))
        
        # Early return if no boxes in the entire batch
        if sum(query_counts) == 0:
            empty_keypoints = torch.zeros((batch_size, 0, self.keypoint_split_projection[0].out_features * len(self.keypoint_split_projection)), device=samples.device)
            return {
                'pred_keypoints': empty_keypoints
            }
        
        # Prepare queries for QFormer (need to handle variable number of queries per image)
        max_queries = max(query_counts)
        padded_queries = torch.zeros((batch_size, max_queries, self.query_embed.weight.shape[1]), device=samples.device)
        padding_mask = torch.ones((batch_size, max_queries), dtype=torch.bool, device=samples.device)
        
        for i, (queries, count) in enumerate(zip(all_queries, query_counts)):
            if count > 0:
                padded_queries[i, :count] = queries
                padding_mask[i, :count] = False
        
        # Forward through the Q-Former with padding mask
        hs, memory = self.qformer(patch_features, padded_queries, padding_mask=padding_mask)
        # hs: [batch_size, max_queries, hidden_dim]
        
        # Predict keypoints (only for non-padded queries)
        shared_features = self.keypoint_share_projection(hs)
        outputs_keypoints = torch.stack([
            keypoint_head(shared_features)
            for keypoint_head in self.keypoint_split_projection
        ], dim=-2).sigmoid() # outputs_keypoints: [B, num_queries, num_bodypoints, 3]
        
        # Reshape keypoints [B, max_queries, num_bodypoints, 3] -> [B, max_queries, num_bodypoints*3]
        outputs_keypoints = outputs_keypoints.reshape(outputs_keypoints.shape[0], outputs_keypoints.shape[1], -1)
        outputs_keypoints = keypoint_xyzxyz_to_xyxyzz(outputs_keypoints)  # outputs_keypoints: [B, max_queries, num_bodypoints*3] xyxyvv
        
        # 패딩된 위치의 값을 0으로 명시적으로 설정
        for i in range(batch_size):
            outputs_keypoints[i, padding_mask[i]] = 0
        
        # Format output
        out = {
            'pred_keypoints': outputs_keypoints, # [B, max_queries, num_bodypoints*3]
            'padding_mask': padding_mask # [B, max_queries]
        }

        return out
"""
targets = [{
            'image_id': torch.tensor([img_id]),
            'boxes': 
            'labels': torch.empty((0,), dtype=torch.int64),
            'keypoints': 
            'area': torch.empty((0,), dtype=torch.float32),
            'iscrowd': torch.empty((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']]),
            'size': torch.as_tensor([img_info['height'], img_info['width']]) : 336*336( 변환 사이즈 )
        }, ..]
        boxes: [N, 4], [cx, cy, w, h], padding, resize, normalize(0-1)
        keypoints: [N, 17*3], xyxyvv, 각 bbox 기준 0-1 정규화
        
samples: [batch_size, channels, height, width]

outputs: 
    out = {
            'pred_keypoints': outputs_keypoints # [B, max_queries, num_bodypoints*3]
            'padding_mask': padding_mask # [B, max_queries]
        }
"""
class SetCriterion(nn.Module):
    """
    Criterion without matcher for keypoint detection.
    """
    def __init__(self, weight_dict, num_body_points):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_body_points = num_body_points
        self.oks = OKSLoss(
            linear=True,
            num_keypoints=num_body_points,
            eps=1e-6,
            reduction='mean',
            loss_weight=1.0
        )
    
    def loss_keypoints(self, outputs, targets, num_boxes):
        """
        Compute the keypoint loss using explicit padding mask.
        
        Args:
            outputs: Model prediction outputs
            targets: Ground truth targets
            padding_mask: Padding mask from the model [batch_size, max_queries]
            num_boxes: Total number of boxes for normalization
        """
        batch_size = len(outputs['pred_keypoints'])
        padding_mask = outputs['padding_mask']
        pred_keypoints_list = []
        target_keypoints_list = []
        # area_list = []
        
        for i in range(batch_size):
            # 현재 이미지의 예측과 타겟
            pred = outputs['pred_keypoints'][i]  # [num_queries, num_keypoints*3]
            tgt = targets[i] 
            
            # 패딩 마스크에서 유효한 쿼리 인덱스 추출 (True가 패딩 위치)
            valid_indices = torch.where(~padding_mask[i])[0]
            
            # 유효한 쿼리가 없거나 타겟 키포인트가 없으면 건너뛰기
            if len(valid_indices) == 0 or len(tgt['keypoints']) == 0:
                continue
            
            # 유효한 예측만 선택
            pred = pred[valid_indices]  # [N, num_keypoints*3]
            
            pred_keypoints_list.append(pred)
            target_keypoints_list.append(tgt['keypoints']) # [N, 17*3], xyxyvv
            
            # # OKS 손실을 위한 영역 정규화
            # max_size = max(tgt['orig_size'][0].item(), tgt['orig_size'][1].item())
            # area = tgt['area'] / (max_size ** 2) # [N]
            # area_list.append(area)
        
        # 처리할 항목이 없으면 0 손실 반환
        device = outputs['pred_keypoints'].device
        if len(pred_keypoints_list) == 0:
            return {
                'loss_keypoints': torch.as_tensor(0., device=device),
                'loss_oks': torch.as_tensor(0., device=device),
            }
        
        # 모든 배치의 예측과 타겟을 합침
        pred_keypoints = torch.cat(pred_keypoints_list, dim=0)  # [total_instances, num_keypoints*3]
        target_keypoints = torch.cat(target_keypoints_list, dim=0)  # [total_instances, num_keypoints*3]
        
        total_instances = pred_keypoints.shape[0]
        area = torch.ones(total_instances, dtype=pred_keypoints.dtype, device=pred_keypoints.device) # [total_instances]
        
        # 키포인트 좌표와 가시성 분리
        Z_pred = pred_keypoints[:, 0:(self.num_body_points * 2)]  # [total_instances, num_keypoints*2]
        Z_gt = target_keypoints[:, 0:(self.num_body_points * 2)]  # [total_instances, num_keypoints*2]
        V_gt = target_keypoints[:, (self.num_body_points * 2):]  # [total_instances, num_keypoints]
        V_pred = pred_keypoints[:, (self.num_body_points * 2):]  # [total_instances, num_keypoints]
        
        # OKS 손실 계산
        oks_loss = self.oks(Z_pred, Z_gt, V_gt, area, weight=None, avg_factor=None, reduction_override=None) # [total_instances]
        
        # L1 손실 계산 (가시성 가중치 적용)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none') # [total_instances, num_keypoints*2]
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        
        # 가시성에 대한 BCE 손실 계산
        vis_loss = F.binary_cross_entropy(
            V_pred, (V_gt > 0).float(), reduction='none'
        ) # [total_instances, num_keypoints] 
        
        losses = {
            'loss_keypoints': pose_loss.sum() / num_boxes,        
            'loss_oks': oks_loss.sum() / num_boxes,
            'loss_visibility': vis_loss.sum() / num_boxes 
        }
        return losses

    def forward(self, outputs, targets):
        """
        전체 손실 계산
        """
        device = next(iter(outputs.values())).device

        # 정규화를 위한 타겟 박스 평균 개수 계산
        num_boxes = sum(len(t["keypoints"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 키포인트 손실 계산
        losses = self.loss_keypoints(outputs, targets, num_boxes)
            
        return losses
class KeypointPostProcessor(nn.Module):
    """
    키포인트 검출 모델의 출력을 후처리하는 모듈
    바운딩 박스 기준 정규화된 키포인트 좌표를 원본 이미지 좌표로 변환합니다.
    패딩 마스크를 활용하여 유효한 예측만 처리합니다.
    """
    def __init__(self, image_input_size=336):
        super().__init__()
        self.image_input_size = image_input_size
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        모델 출력을 후처리하여 최종 검출 결과를 얻습니다.

        Args:
            outpits = {
            'pred_keypoints': outputs_keypoints # [B, max_queries, num_bodypoints*3] 0-1 정규화
            'padding_mask': padding_mask # [B, max_queries]
        }
        targets = [{
            'image_id': torch.tensor([img_id]),
            'boxes': 
            'labels': torch.empty((0,), dtype=torch.int64),
            'keypoints': 
            'area': torch.empty((0,), dtype=torch.float32),
            'iscrowd': torch.empty((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']]),
            'size': torch.as_tensor([img_info['height'], img_info['width']]) : 336*336( 변환 사이즈 )
        }, ..]
        boxes: [N, 4], [cx, cy, w, h], padding, resize, normalize(0-1)
        keypoints: [N, 17*3], xyxyvv, 각 bbox 기준 0-1 정규화

        Returns:
            results: 처리된 검출 결과 리스트
        """
        from util.box_ops import box_cxcywh_to_xyxy
        
        out_keypoints = outputs['pred_keypoints']
        padding_mask = outputs['padding_mask']
        batch_size = len(out_keypoints)

        # 각 이미지에 대해 처리
        results = []
        for i in range(batch_size):
            # 현재 이미지의 타겟과 예측된 키포인트
            target = targets[i]
            keypoints = out_keypoints[i]
            
            # 패딩 마스크에서 유효한 쿼리 인덱스 추출 (True가 패딩 위치)
            valid_indices = torch.where(~padding_mask[i])[0]
            
            # 유효한 예측이 없는 경우 빈 결과 추가
            if len(valid_indices) == 0:
                results.append({
                    'keypoints': torch.zeros((0, keypoints.shape[-1]), device=keypoints.device)
                })
                continue
                
            # 유효한 예측만 선택
            keypoints = keypoints[valid_indices]
            
            # 원본 이미지 크기 정보
            img_h, img_w = target['orig_size']
            boxes = target['boxes']  # [N, 4] 형식, [cx, cy, w, h], 정규화됨
            
            # 패딩 정보 계산
            max_size = max(img_h.item(), img_w.item())
            left_pad = (max_size - img_w.item()) // 2
            top_pad = (max_size - img_h.item()) // 2
                        
            # 결과 저장을 위한 텐서
            num_valid = len(valid_indices)
            adjusted_keypoints = torch.zeros_like(keypoints)
            
            # 바운딩 박스를 [cx, cy, w, h]에서 [x1, y1, x2, y2]로 변환
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            
            # 바운딩 박스별로 키포인트 좌표 변환
            for k in range(num_valid):
                # 바운딩 박스 좌표 ([x1, y1, x2, y2] 형식)
                x1, y1, x2, y2 = boxes_xyxy[k]
                
                # 역스케일링
                x1 = x1 * max_size
                y1 = y1 * max_size
                x2 = x2 * max_size
                y2 = y2 * max_size

                # 패딩 제거
                x1 = x1 - left_pad
                y1 = y1 - top_pad
                x2 = x2 - left_pad
                y2 = y2 - top_pad
                
                # 이미지 경계 내로 클리핑
                x1 = torch.clamp(x1, 0, img_w)
                y1 = torch.clamp(y1, 0, img_h)
                x2 = torch.clamp(x2, 0, img_w)
                y2 = torch.clamp(y2, 0, img_h)
                
                # 바운딩 박스 크기 계산 (원본 이미지 기준)
                box_w = x2 - x1
                box_h = y2 - y1
                
                # 키포인트 좌표 형식: [xyxy...zz...]
                num_keypoints = keypoints.shape[-1] // 3
                
                # 키포인트 좌표 변환 (바운딩 박스 기준 상대좌표 -> 이미지 좌표)
                for j in range(num_keypoints):
                    # x, y 좌표 인덱스
                    idx_x = j*2
                    idx_y = j*2 + 1
                    
                    # 바운딩 박스 기준의 상대좌표를 절대좌표로 변환
                    adjusted_keypoints[k, idx_x] = keypoints[k, idx_x] * box_w + x1
                    adjusted_keypoints[k, idx_y] = keypoints[k, idx_y] * box_h + y1
                    
                    # 이미지 경계를 벗어나지 않도록 클리핑
                    adjusted_keypoints[k, idx_x] = torch.clamp(adjusted_keypoints[k, idx_x], 0, img_w)
                    adjusted_keypoints[k, idx_y] = torch.clamp(adjusted_keypoints[k, idx_y], 0, img_h)
                
                # 가시성 값은 그대로 복사 (마지막 third*3 부분)
                adjusted_keypoints[k, num_keypoints * 2:] = keypoints[k, num_keypoints * 2:]
                
            # 결과 저장
            results.append({
                'keypoints': adjusted_keypoints,  # [num_valid, num_bodypoints*3] xyxyvv, v is 0-1
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
    # 손실 가중치 설정
    weight_dict = {
        "loss_keypoints": args.keypoints_loss_coef,
        "loss_oks": args.oks_loss_coef,
        "loss_visibility": args.visibility_loss_coef
    }
    
    # 모델 생성
    model = CLIPDETR(
        clip_model_name=args.clip_model_name,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
        num_body_points=args.num_body_points,
    )
    
    # Criterion 생성 (매처 없음)
    criterion = SetCriterion(
        weight_dict=weight_dict,
        num_body_points=args.num_body_points
    )
    
    # 후처리기 설정
    postprocessors = {
        'keypoints': KeypointPostProcessor(image_input_size=args.image_input_size)
    }
    
    return model, criterion, postprocessors