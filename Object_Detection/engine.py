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
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import wandb
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, accumulation_steps: int = 16, start_step: int = 0):
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
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Calculate total loss based on loss_dict and weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Normalize loss by accumulation steps to maintain scale
        losses = losses / accumulation_steps
        
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
                    "train/loss_ce": loss_dict_reduced['loss_ce'] * weight_dict['loss_ce'],
                    "train/loss_bbox": loss_dict_reduced['loss_bbox'] * weight_dict['loss_bbox'] if 'loss_bbox' in loss_dict_reduced else 0,
                    "train/loss_giou": loss_dict_reduced['loss_giou'] * weight_dict['loss_giou'] if 'loss_giou' in loss_dict_reduced else 0,
                    "train/class_error": loss_dict_reduced['class_error'],
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
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


class DetectionEvaluator:
    """
    객체 검출 모델의 성능을 평가하기 위한 커스텀 평가 도구
    COCO API 없이 AP, AR 등의 지표를 계산합니다.
    """
    def __init__(self):
        # 클래스별 예측 결과 저장 딕셔너리
        self.predictions = defaultdict(list)
        # 클래스별 정답 저장 딕셔너리
        self.ground_truths = defaultdict(list)
        # 이미 처리한 이미지 ID 목록
        self.processed_img_ids = set()
        # IoU 임계값 (COCO와 유사하게 설정)
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
        # 면적 범주 (small, medium, large)
        self.area_ranges = {
            'all': (0, float('inf')),
            'small': (0, 32**2),
            'medium': (32**2, 96**2),
            'large': (96**2, float('inf'))
        }
        
    def update(self, predictions, targets, image_input_size):
        """
        예측 결과와 정답을 업데이트합니다.
        
        Args:
            predictions: 모델의 예측 결과 딕셔너리 (이미지 ID를 키로 가짐)
            targets: 정답 데이터 리스트
        """
        for img_id, pred in predictions.items():
            if img_id in self.processed_img_ids:
                continue
                
            self.processed_img_ids.add(img_id)
            
            # 현재 이미지에 대한 정답 찾기
            gt = next((t for t in targets if t['image_id'].item() == img_id), None)
            if gt is None:
                continue
                
            # 예측 결과 저장
            for score, label, box in zip(pred['scores'], pred['labels'], pred['boxes']):
                self.predictions[label.item()].append({
                    'image_id': img_id,
                    'bbox': box.tolist(),
                    'score': score.item()
                })
            
            # 정답 저장
            gt_boxes = gt['boxes']  # 정규화된 cxcywh 형식
            gt_labels = gt['labels']
            gt_areas = gt['area']
            
            # 원본 이미지 크기 정보
            img_h, img_w = gt['orig_size'].tolist()
            
            # 패딩 정보 계산
            max_size = max(img_h, img_w)
            left_pad = (max_size - img_w) // 2
            top_pad = (max_size - img_h) // 2
            scale = image_input_size / max_size  # 모델 입력 크기 기준 스케일
            
            # GT 박스 좌표 변환 (cxcywh -> xyxy, 패딩 및 스케일링 고려)
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)  # 정규화된 xyxy
            
            # 정규화된 좌표를 모델 입력 크기로 변환
            gt_boxes_xyxy = gt_boxes_xyxy * image_input_size
            
            # 336 -> 원본 정사각형 크기로 역스케일링
            gt_boxes_xyxy = gt_boxes_xyxy / scale
            
            # 패딩 제거
            gt_boxes_adj = []
            for box in gt_boxes_xyxy:
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, x1 - left_pad)
                x2 = min(img_w, x2 - left_pad)
                y1 = max(0, y1 - top_pad)
                y2 = min(img_h, y2 - top_pad)
                gt_boxes_adj.append([x1, y1, x2, y2])
            
            gt_boxes_adj = torch.tensor(gt_boxes_adj, device=gt_boxes.device)
            
            # 정답 저장 (조정된 xyxy 형식)
            for label, box, area in zip(gt_labels, gt_boxes_adj, gt_areas):
                self.ground_truths[label.item()].append({
                    'image_id': img_id,
                    'bbox': box.tolist(),
                    'area': area.item(),
                    'used': False  # TP/FP 판단 시 사용 여부
                })
                
                
    def synchronize_between_processes(self):
        """
        분산 훈련 환경에서 여러 프로세스 간 결과를 동기화합니다.
        """
        all_predictions = utils.all_gather(self.predictions)
        all_ground_truths = utils.all_gather(self.ground_truths)
        
        # 결과 병합
        merged_predictions = defaultdict(list)
        merged_ground_truths = defaultdict(list)
        
        for predictions in all_predictions:
            for label, preds in predictions.items():
                merged_predictions[label].extend(preds)
                
        for ground_truths in all_ground_truths:
            for label, gts in ground_truths.items():
                merged_ground_truths[label].extend(gts)
        
        self.predictions = merged_predictions
        self.ground_truths = merged_ground_truths
    
    def calculate_iou(self, box1, box2):
        """
        두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
        
        Args:
            box1, box2: [x1, y1, x2, y2] 형식의 박스 좌표
            
        Returns:
            iou: IoU 값
        """
        # 박스1
        x1_1, y1_1, x2_1, y2_1 = box1
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        
        # 박스2
        x1_2, y1_2, x2_2, y2_2 = box2
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        
        # 두 박스의 면적
        area1, area2 = w1 * h1, w2 * h2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # 교집합이 없는 경우
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        # 교집합 면적
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # IoU = 교집합 / 합집합
        union = area1 + area2 - intersection
        iou = intersection / union
        
        return iou
    
    def calculate_ap(self, precision, recall):
        """
        정밀도-재현율 곡선의 면적(AP)을 계산합니다.
        COCO 스타일의 101-포인트 보간법을 사용합니다.
        
        Args:
            precision, recall: 정밀도와 재현율 배열
            
        Returns:
            ap: Average Precision
        """
        # COCO 스타일의 101-포인트 보간 AP 계산
        all_points = np.linspace(0, 1, 101)  # 0.0, 0.01, 0.02, ..., 1.0
        mprecision = np.concatenate(([0], precision, [0]))
        mrecall = np.concatenate(([0], recall, [1]))
        
        # 뒤에서부터 최대값으로 precision 보정
        for i in range(mprecision.size - 2, -1, -1):
            mprecision[i] = max(mprecision[i], mprecision[i + 1])
            
        # recall 변화 지점 찾기
        i_list = []
        for i in range(1, mrecall.size):
            if mrecall[i] != mrecall[i - 1]:
                i_list.append(i)
                
        # AP 계산
        ap = 0
        for i in i_list:
            ap += ((mrecall[i] - mrecall[i - 1]) * mprecision[i])
            
        return ap
    
    def evaluate_category(self, category_id, iou_threshold, area_range):
        """
        특정 카테고리, IoU 임계값, 면적 범위에 대한 평가를 수행합니다.
        
        Args:
            category_id: 평가할 카테고리 ID
            iou_threshold: IoU 임계값
            area_range: 면적 범위 (min_area, max_area)
            
        Returns:
            ap: Average Precision
            precisions, recalls: 정밀도와 재현율 배열
            scores: 점수 배열
            tp, fp: True Positive와 False Positive 배열
        """
        min_area, max_area = area_range
        
        # 해당 카테고리의 예측 결과와 정답
        predictions = self.predictions.get(category_id, [])
        ground_truths = self.ground_truths.get(category_id, [])
        
        # 예측 결과가 없으면 0 반환
        if not predictions:
            return 0.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
        # 점수 기준 내림차순 정렬
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # 이미지별 정답 인덱스 맵 생성
        gt_by_img = defaultdict(list)
        for i, gt in enumerate(ground_truths):
            if min_area <= gt['area'] <= max_area:
                gt_by_img[gt['image_id']].append(i)
                
        # 평가용 배열 초기화
        n_predictions = len(predictions)
        tp = np.zeros(n_predictions)
        fp = np.zeros(n_predictions)
        gt_used = {i: False for i in range(len(ground_truths))}
        scores = np.zeros(n_predictions)
        
        # 각 예측에 대해 평가
        for i, pred in enumerate(predictions):
            scores[i] = pred['score']
            img_id = pred['image_id']
            pred_bbox = pred['bbox']
            
            # 해당 이미지의 정답이 없으면 FP
            if img_id not in gt_by_img or not gt_by_img[img_id]:
                fp[i] = 1
                continue
                
            # 해당 이미지의 모든 정답과 IoU 계산
            max_iou = -1
            max_gt_idx = -1
            
            for gt_idx in gt_by_img[img_id]:
                if gt_used[gt_idx]:
                    continue
                    
                gt_bbox = ground_truths[gt_idx]['bbox']
                iou = self.calculate_iou(pred_bbox, gt_bbox)
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # IoU 임계값 이상이면 TP, 아니면 FP
            if max_iou >= iou_threshold and max_gt_idx >= 0:
                gt_used[max_gt_idx] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # 비어있는 경우 처리
        if n_predictions == 0:
            return 0.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
        # 누적합 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 정밀도와 재현율 계산
        n_gt = sum(1 for gt in ground_truths if min_area <= gt['area'] <= max_area)
        recalls = tp_cumsum / n_gt if n_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # AP 계산
        ap = self.calculate_ap(precisions, recalls)
        
        return ap, precisions, recalls, scores, tp, fp
    
    def evaluate(self):
        """
        전체 평가를 수행하고 결과를 반환합니다.
        
        Returns:
            stats: 다양한 조건에서의 평가 지표
        """
        # 모든 카테고리 ID
        categories = sorted(set(list(self.predictions.keys()) + list(self.ground_truths.keys())))
        
        # 결과 저장을 위한 딕셔너리
        stats = {}
        
        # AP 계산 (다양한 IoU 임계값과 면적 범위)
        ap_per_category = defaultdict(list)
        
        for iou_threshold in self.iou_thresholds:
            for area_name, area_range in self.area_ranges.items():
                ap_list = []
                
                for category_id in categories:
                    ap, _, _, _, _, _ = self.evaluate_category(category_id, iou_threshold, area_range)
                    ap_per_category[category_id].append(ap)
                    ap_list.append(ap)
                
                # 평균 AP 계산
                if ap_list:
                    mAP = np.mean(ap_list)
                else:
                    mAP = 0.0
                
                # 결과 저장
                key = f'AP_{int(iou_threshold * 100):02d}_{area_name}'
                stats[key] = mAP
        
        # AP@.50 (PASCAL VOC 스타일)
        stats['AP50'] = stats.get('AP_50_all', 0.0)
        # AP@.75
        stats['AP75'] = stats.get('AP_75_all', 0.0)
        # mAP (모든 IoU 임계값의 평균)
        stats['mAP'] = np.mean([stats[f'AP_{int(iou * 100):02d}_all'] for iou in self.iou_thresholds])
        # 면적별 mAP
        stats['mAP_small'] = np.mean([stats[f'AP_{int(iou * 100):02d}_small'] for iou in self.iou_thresholds])
        stats['mAP_medium'] = np.mean([stats[f'AP_{int(iou * 100):02d}_medium'] for iou in self.iou_thresholds])
        stats['mAP_large'] = np.mean([stats[f'AP_{int(iou * 100):02d}_large'] for iou in self.iou_thresholds])
        
        # 카테고리별 평균 AP
        for category_id in categories:
            stats[f'AP_category_{category_id}'] = np.mean(ap_per_category[category_id])
        
        # COCO 형식과 호환되는 배열 생성
        coco_stats = np.zeros(12)
        coco_stats[0] = stats['mAP']  # AP @[.5:.95]
        coco_stats[1] = stats['AP50']  # AP @.50
        coco_stats[2] = stats['AP75']  # AP @.75
        coco_stats[3] = stats['mAP_small']  # AP small
        coco_stats[4] = stats['mAP_medium']  # AP medium
        coco_stats[5] = stats['mAP_large']  # AP large
        # 6-11은 AR 지표를 위한 공간이지만 현재 구현에서는 계산하지 않음
        
        stats['stats'] = coco_stats
        
        return stats
    
    # def accumulate(self):
    #     """
    #     결과를 누적하는 메서드 (COCO API와의 호환성을 위해 유지)
    #     """
    #     pass
    
    def summarize(self):
        """
        결과를 요약하여 출력합니다.
        """
        stats = self.evaluate()
        
        print('Detection evaluation results:')
        print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=all   ] = {stats['mAP']:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.50      | area=all   ] = {stats['AP50']:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.75      | area=all   ] = {stats['AP75']:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=small ] = {stats['mAP_small']:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium] = {stats['mAP_medium']:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=large ] = {stats['mAP_large']:.3f}")
        
        return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir=None):
    """
    COCO API 없이 객체 검출 모델의 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        criterion: 손실 함수
        postprocessors: 모델 출력 후처리 함수
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
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    # 커스텀 평가 도구 초기화
    evaluator = DetectionEvaluator()
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                   for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                     for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        # 원본 이미지 크기로 결과 변환
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # 평가용 딕셔너리 생성
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        # 평가 객체 업데이트
        image_input_size = postprocessors['bbox'].image_input_size
        evaluator.update(res, targets, image_input_size)
    
    # 프로세스 간 결과 동기화
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluator.synchronize_between_processes()
    
    # 평가 수행
    # evaluator.accumulate()
    eval_stats = evaluator.summarize()
    
    # 결과 수집
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['coco_eval_bbox'] = eval_stats['stats'].tolist()
    
    return stats, evaluator