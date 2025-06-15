"""
CLIP-DETR 모델의 추론 스크립트

이 스크립트는 저장된 체크포인트를 불러와 검증 데이터셋에서 추론 및 평가를 수행합니다.
또한 선택적으로 예측 결과를 시각화할 수 있습니다.
"""

import argparse
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

from models.clip_detr import build_model
from datasets import build_dataset
from engine import evaluate
from util.box_ops import box_cxcywh_to_xyxy
import util
from torch.utils.data import DataLoader
from util.misc import collate_fn
# COCO 클래스 매핑 (id -> 클래스명)
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 
    90: "toothbrush"
}

# 랜덤한 색상 생성
def generate_colors(n):
    random.seed(42)  # 재현성을 위한 시드 설정
    colors = []
    for _ in range(n):
        color = tuple(random.uniform(0, 1) for _ in range(3))
        colors.append(color)
    return colors

# 클래스별 색상 맵 생성
COLORS = generate_colors(len(COCO_CLASSES))
CLASS_COLORS = {class_id: COLORS[i % len(COLORS)] for i, class_id in enumerate(COCO_CLASSES.keys())}

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-DETR Inference', add_help=False)
    
    # 모델 매개변수
    parser.add_argument('--clip_model_name', default='openai/clip-vit-large-patch14-336', type=str,
                        help='Name of CLIP model to use')
    parser.add_argument('--num_classes', default=91, type=int, help="Number of classes")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Dimension of feedforward")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dataset_file', default='coco')
    
    # Hungarian Matcher 매개변수
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # 손실 가중치
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    # 데이터셋 관련
    parser.add_argument('--coco_path', type=str, default='./data/coco', help='COCO dataset path')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    
    # 추론 관련
    parser.add_argument('--device', default='cuda', help='device to use for evaluation')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to the checkpoint')
    parser.add_argument('--output_dir', default='./inference_results', help='path to save results')
    
    # 시각화 관련
    parser.add_argument('--image_input_size', default=336, type=int) # clip model input image size
    parser.add_argument('--visualize', action='store_true', help='whether to visualize predictions')
    parser.add_argument('--num_vis_images', default=5, type=int, help='number of images to visualize')
    parser.add_argument('--threshold', default=0.5, type=float, help='confidence threshold for visualization')
    
    # 분산 훈련 매개변수
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser

def visualize_prediction(image, outputs, threshold=0.5, output_path=None):
    """
    이미지와 모델 예측 결과를 시각화합니다.
    
    Args:
        image: 원본 PIL 이미지
        outputs: 모델 예측 결과 (boxes, scores, labels)
        threshold: 점수 임계값
        output_path: 결과 저장 경로
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # 예측 결과 추출
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    
    # 임계값 이상의 예측만 시각화
    valid_indices = scores >= threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    
    # 각 예측에 대한 bounding box와 라벨 표시
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # 클래스 이름 가져오기
        class_name = COCO_CLASSES.get(label, f"Unknown ({label})")
        color = CLASS_COLORS.get(label, (1, 0, 0))  # 기본 색상은 빨간색
        
        # Bounding box 그리기
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 라벨과 점수 표시
        ax.text(
            x1, y1 - 5, f"{class_name}: {score:.2f}", 
            color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.7)
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    # 결과 저장
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {output_path}")
    
    plt.close(fig)
    return fig

def main(args):
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device(args.device)
    
    # 랜덤 시드 설정
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 모델 초기화 및 체크포인트 로드
    print("모델 로드 중...")
    model, criterion, postprocessors = build_model(args)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # 데이터셋 초기화 (데이터로더는 시각화에만 필요하면 생성하지 않음)
    print("데이터셋 로드 중...")
    dataset_val = build_dataset(image_set='val', args=args)
    
    # 시각화만 수행하는 경우
    if args.visualize:
        vis_dir = output_dir
        
        print("\n예측 결과 시각화 중...")
        
        # 무작위로 이미지 선택
        indices = random.sample(range(len(dataset_val)), min(args.num_vis_images, len(dataset_val)))
        
        for i, idx in enumerate(indices):
            # 이미지와 타겟 불러오기
            img, target = dataset_val[idx]
            
            # 원본 이미지 경로 가져오기
            img_id = target['image_id'].item()
            
            # 원본 이미지 파일명 찾기
            file_name = None
            for img_info in dataset_val.coco_json['images']:
                if img_info['id'] == img_id:
                    file_name = img_info['file_name']
                    break
            
            if file_name is None:
                print(f"이미지 ID {img_id}에 대한 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            img_path = os.path.join(args.coco_path, 'val2017', file_name)
            
            # 원본 이미지 로드
            original_img = Image.open(img_path).convert('RGB')
            w, h = original_img.size
            
            # 이미지를 모델에 입력하기 위해 배치로 변환
            img = img.unsqueeze(0).to(device)
            
            # 추론 실행
            with torch.no_grad():
                outputs = model(img)
                
                # 포스트프로세싱 - 정규화된 좌표를 픽셀 좌표로 변환
                target_sizes = torch.tensor([[h, w]]).to(device)
                results = postprocessors['bbox'](outputs, target_sizes)
                result = results[0]  # 배치 크기 1
                
            # 예측 결과 시각화
            output_path = vis_dir / f"pred_{i:03d}_{img_id}.png"
            visualize_prediction(original_img, result, args.threshold, output_path)
            print(f"처리 완료: 이미지 {i+1}/{len(indices)} (ID: {img_id})")
        
        print(f"시각화 완료. 결과는 {vis_dir} 디렉토리에 저장되었습니다.")
    else:
        # 평가 실행 (원하는 경우에만)
        print("평가 중...")
        # 데이터로더 생성
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(
            dataset_val, args.batch_size, sampler=sampler_val,
            drop_last=False, num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate_fn
        )
        
        stats, evaluator = evaluate(model, criterion, postprocessors, data_loader_val, device, output_dir)
        
        # 결과 출력
        print("\n평가 결과:")
        print(f"mAP: {stats['coco_eval_bbox'][0]:.3f}")
        print(f"mAP@50: {stats['coco_eval_bbox'][1]:.3f}")
        print(f"mAP@75: {stats['coco_eval_bbox'][2]:.3f}")
        
        # 결과 파일 저장
        result_path = output_dir / "eval_results.txt"
        with open(result_path, 'w') as f:
            f.write(f"mAP: {stats['coco_eval_bbox'][0]:.6f}\n")
            f.write(f"mAP@50: {stats['coco_eval_bbox'][1]:.6f}\n")
            f.write(f"mAP@75: {stats['coco_eval_bbox'][2]:.6f}\n")
            f.write(f"mAP_small: {stats['coco_eval_bbox'][3]:.6f}\n")
            f.write(f"mAP_medium: {stats['coco_eval_bbox'][4]:.6f}\n")
            f.write(f"mAP_large: {stats['coco_eval_bbox'][5]:.6f}\n")
            
        print(f"평가 결과가 {result_path}에 저장되었습니다.")
    
    print("추론 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CLIP-DETR Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)