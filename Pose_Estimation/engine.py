"""
Training and evaluation functions for CLIP-DETR.

This file contains standard training and evaluation loops and doesn't need significant changes.
"""

import math
import os
import sys
import torch
import util.misc as utils
from typing import Iterable
import wandb
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    accumulation_steps: int = 16, start_step: int = 0):
    """
    Train the model for one epoch with gradient accumulation.
    
    Args:
        model: Model to train
        criterion: Loss criterion
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        max_norm: Maximum norm for gradient clipping
        accumulation_steps: Number of steps to accumulate gradients (default: 4)
    """
    model.train()
    # Print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    
    # Set initial step for gradient accumulation
    step = 0
    global_step = start_step
    # Zero the gradients at the beginning
    optimizer.zero_grad()
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step += 1
        samples = samples.to(device) # 원본은 nestedtensor로 들어온다. 지금은 그냥 tensor
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Calculate total loss based on loss_dict and weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Normalize loss by accumulation steps to maintain scale
        losses = losses / accumulation_steps
        # Print the loss value for debugging
        # Backward pass
        losses.backward()
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        # Only update weights after accumulation_steps iterations or at the end of the epoch
        if step % accumulation_steps == 0 or step == len(data_loader):
            # Apply gradient clipping if max_norm > 0
            global_step += 1
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            # For logging, we need the unscaled loss value (multiply back by accumulation_steps)
            loss_value_for_logging = losses.item() * accumulation_steps
            
            # W&B 로깅 - 누적 그래디언트 스텝마다
            if utils.is_main_process():
                wandb.log({
                    "train/step": global_step,
                    "train/epoch": epoch,
                    "train/loss": loss_value_for_logging,
                    "train/loss_keypoints": loss_dict_reduced['loss_keypoints'] * weight_dict['loss_keypoints'] if 'loss_keypoints' in loss_dict_reduced else 0,
                    "train/loss_oks": loss_dict_reduced['loss_oks'] * weight_dict['loss_oks'] if 'loss_oks' in loss_dict_reduced else 0,
                    "train/loss_visibility": loss_dict_reduced['loss_visibility'] * weight_dict['loss_visibility'] if 'loss_visibility' in loss_dict_reduced else 0,
                    "train/lr": optimizer.param_groups[0]["lr"]
                })
        else:
            # Use loss value only for logging
            with torch.no_grad():
                loss_value_for_logging = losses.item() * accumulation_steps
        
        
        # Check if loss is finite
        if not math.isfinite(loss_value_for_logging):
            print(f"Loss is {loss_value_for_logging}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        # Update metric logger
        metric_logger.update(loss=loss_value_for_logging, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


import torch
import numpy as np
import math
from collections import defaultdict
import util.misc as utils

class SimplifiedEvaluator:
    """
    키포인트 검출 모델을 위한 단순화된 평가기
    다양한 OKS 임계값에 따른 정확도를 계산합니다.
    """
    def __init__(self, thresholds=[0.5, 0.7, 0.9]):
        self.thresholds = thresholds
        self.all_oks_values = []      # 모든 OKS 값
        self.all_losses = []          # 모든 손실 값
        
    def update(self, outputs, targets, loss=None):
        """
        배치 결과로 평가 지표를 업데이트합니다.
        outputs의 padding_mask를 활용하여 유효한 예측만 추출합니다.
        
        Args:
            outputs: 모델의 원시 출력값
            targets: 정답 데이터 리스트
            loss: 현재 배치의 손실 값 (선택 사항)
        """
        # 손실 값이 제공된 경우 저장
        if loss is not None:
            self.all_losses.append(loss)
            
        batch_size = len(outputs['pred_keypoints'])
        if 'padding_mask' not in outputs:
            return
            
        padding_mask = outputs['padding_mask']
        
        for i in range(batch_size):
            # 현재 이미지의 예측과 타겟
            pred = outputs['pred_keypoints'][i]  # [num_queries, num_keypoints*3]
            tgt = targets[i]
            
            # 타겟에 keypoints가 없는 경우 건너뛰기
            if 'keypoints' not in tgt or tgt['keypoints'].shape[0] == 0:
                continue
                
            # 패딩 마스크에서 유효한 쿼리 인덱스 추출 (True가 패딩 위치)
            valid_indices = torch.where(~padding_mask[i])[0]
            
            # 유효한 쿼리가 없으면 건너뛰기
            if len(valid_indices) == 0:
                continue
            
            # 유효한 예측만 선택
            valid_pred = pred[valid_indices]  # [N, num_keypoints*3]
            valid_gt = tgt['keypoints']      # [M, num_keypoints*3]
            
            # GT와 예측의 수가 일치하는지 확인
            # 참고: 실제 데이터에서는 일치해야 함
            num_valid = min(len(valid_pred), len(valid_gt))
            
            # 같은 인덱스끼리 직접 매칭
            for j in range(num_valid):
                # OKS 계산
                oks = self.calculate_oks(valid_pred[j], valid_gt[j])
                self.all_oks_values.append(oks)
            
    def synchronize_between_processes(self):
        """
        분산 훈련 환경에서 여러 프로세스 간 결과를 동기화합니다.
        """
        all_oks = utils.all_gather(self.all_oks_values)
        all_losses = utils.all_gather(self.all_losses)
        
        # 결과 병합
        merged_oks = []
        merged_losses = []
        
        for oks in all_oks:
            merged_oks.extend(oks)
            
        for losses in all_losses:
            merged_losses.extend(losses)
        
        self.all_oks_values = merged_oks
        self.all_losses = merged_losses
    
    def evaluate(self):
        """
        전체 평가 데이터에 대한 평가 지표를 계산합니다.
        
        Returns:
            stats: 평가 지표 딕셔너리
        """
        if not self.all_oks_values:
            return {
                'loss': np.mean(self.all_losses) if self.all_losses else 0.0,
                'oks': 0.0,
                'accuracy': 0.0,
                'accuracy_50': 0.0,
                'accuracy_70': 0.0,
                'accuracy_90': 0.0,
                'coco_eval_keypoints': [0.0, 0.0, 0.0, 0.0] # [accuracy_50, accuracy_70, accuracy_90, avg_oks]
            }
        
        # 평균 OKS 계산
        avg_oks = np.mean(self.all_oks_values)
        
        # 각 임계값에 대한 정확도 계산
        accuracies = {}
        for threshold in self.thresholds:
            accuracy = np.mean([1 if oks >= threshold else 0 for oks in self.all_oks_values])
            accuracies[f'accuracy_{int(threshold*100)}'] = accuracy
        
        # 평균 손실 계산
        avg_loss = np.mean(self.all_losses) if self.all_losses else 0.0
        
        # 결과 반환
        stats = {
            'loss': avg_loss,
            'oks': avg_oks,
            'accuracy': accuracies.get('accuracy_50', 0.0),  # 기본값은 OKS 0.5에서의 정확도
        }
        
        # 각 임계값별 정확도 추가
        stats.update(accuracies)
        
        # main.py 호환을 위한 형식 유지
        # [accuracy_50, accuracy_70, accuracy_90, avg_oks]
        stats['coco_eval_keypoints'] = [
            accuracies.get('accuracy_50', 0.0),
            accuracies.get('accuracy_70', 0.0),
            accuracies.get('accuracy_90', 0.0),
            avg_oks
        ]
        
        return stats
        
    def calculate_oks(self, pred_keypoints, gt_keypoints):
        """
        Object Keypoint Similarity (OKS) 계산
        
        Args:
            pred_keypoints: 예측 키포인트 텐서 [num_keypoints*3]
            gt_keypoints: GT 키포인트 텐서 [num_keypoints*3]
            
        Returns:
            oks: OKS 값 (0~1 사이)
        """
        # xyxyvv 형식 가정 (x좌표들, y좌표들, 가시성 값들)
        num_keypoints = gt_keypoints.shape[0] // 3
        
        # 좌표와 가시성 분리
        x_pred = pred_keypoints[:num_keypoints*2:2]  # x 좌표들
        y_pred = pred_keypoints[1:num_keypoints*2:2]  # y 좌표들
        
        x_gt = gt_keypoints[:num_keypoints*2:2]  # x 좌표들
        y_gt = gt_keypoints[1:num_keypoints*2:2]  # y 좌표들
        v_gt = gt_keypoints[num_keypoints*2:]  # 가시성 값들
        
        # COCO 키포인트별 표준 sigma 값 (키포인트 유형별 정확도 요구사항)
        sigmas = torch.tensor([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
        ], device=pred_keypoints.device) / 10.0
        
        # 키포인트 수에 맞게 sigmas 조정
        if len(sigmas) != num_keypoints:
            sigmas = torch.ones(num_keypoints, device=pred_keypoints.device) * 0.05
            
        # 거리 계산
        dx = x_pred - x_gt
        dy = y_pred - y_gt
        d = (dx**2 + dy**2) / (2 * (sigmas**2) * 1.0)  # area=1.0으로 통일
        
        # OKS 계산: exp(-d/2) * visibility
        oks_per_keypoint = torch.exp(-d) * (v_gt > 0).float()
        valid_keypoints = (v_gt > 0).float().sum()
        
        if valid_keypoints > 0:
            oks = oks_per_keypoint.sum() / valid_keypoints
        else:
            oks = torch.tensor(0.0, device=pred_keypoints.device)
            
        return oks.item()
    
    def summarize(self):
        """
        평가 결과를 출력합니다.
        """
        stats = self.evaluate()
        
        print('Keypoint detection evaluation results (Simplified):')
        print(f"Average Loss: {stats['loss']:.4f}")
        print(f"Average OKS: {stats['oks']:.4f}")
        
        # 각 임계값별 정확도 출력
        for threshold in self.thresholds:
            accuracy_key = f'accuracy_{int(threshold*100)}'
            print(f"Accuracy @ OKS={threshold:.1f}: {stats[accuracy_key]:.4f}")
        
        return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir=None):
    """
    키포인트 검출 모델의 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        criterion: 손실 함수
        postprocessors: 모델 출력 후처리 함수 (사용하지 않음)
        data_loader: 평가 데이터 로더
        device: 평가를 수행할 장치
        output_dir: 결과를 저장할 디렉터리 (선택 사항)
        
    Returns:
        stats: 평가 지표
        evaluator: 평가 객체
    """
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # 평가기 초기화 - OKS 임계값 [0.5, 0.7, 0.9] 사용
    evaluator = SimplifiedEvaluator(thresholds=[0.5, 0.7, 0.9])
    
    # 모든 평가 데이터에 대해 추론
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 모델 추론
        outputs = model(samples, targets)  # outputs에 padding_mask가 포함됨
        
        # 손실 계산
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # 총 손실 계산
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 손실값 집계
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                   for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                     for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=total_loss.item(),
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled)
        
        # 후처리 없이 바로 모델 출력을 평가에 사용
        # padding_mask를 활용하여 유효한 예측만 추출하고 같은 인덱스끼리 매칭
        evaluator.update(outputs, targets, total_loss.item())
    
    # 프로세스 간 결과 동기화
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluator.synchronize_between_processes()
    
    # 평가 결과 계산
    eval_stats = evaluator.summarize()
    
    # 결과 수집
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 실제 계산된 metrics 추가
    stats.update(eval_stats)
    
    # COCO 호환성을 위한 형식
    # main.py에서 참조하는 형식대로 유지
    # stats['coco_eval_keypoints'] = eval_stats['coco_eval_keypoints']
    
    """
    stats = {
    'loss': avg_loss,
    'oks': avg_oks,
    'accuracy': accuracies.get('accuracy_50', 0.0),  # 기본값은 OKS 0.5에서의 정확도
    'accuracy_50': accuracies.get('accuracy_50', 0.0),
    'accuracy_70': accuracies.get('accuracy_70', 0.0),
    'accuracy_90': accuracies.get('accuracy_90', 0.0),
    'coco_eval_keypoints': [
        accuracies.get('accuracy_50', 0.0),
        accuracies.get('accuracy_70', 0.0),
        accuracies.get('accuracy_90', 0.0),
        avg_oks
    ]
}
    
    """
    
    return stats, evaluator