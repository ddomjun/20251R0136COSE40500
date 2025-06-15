"""
Inference and evaluation script for CLIP-DETR pose estimation.

This script loads a trained CLIP-DETR model and:
- With the --visualize flag: Runs inference on a subset of COCO validation set and visualizes results
- Without the --visualize flag: Runs full evaluation on COCO validation set

Example usage for visualization:
    python inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir /path/to/output \
        --coco_path /path/to/coco \
        --num_vis_images 20 \
        --threshold 0.7 \
        --visualize

Example usage for evaluation:
    python inference.py \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir /path/to/output \
        --coco_path /path/to/coco
"""

import argparse
import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms as T

import util.misc as utils
from datasets import build_dataset
from models.clip_detr import build_model
from engine import evaluate


# COCO keypoint names and skeleton for visualization
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for visualization (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Colors for visualization
KEYPOINT_COLORS = [
    (255, 0, 0),    # Nose - Red
    (255, 85, 0),   # Left eye - Orange
    (255, 170, 0),  # Right eye - Orange
    (255, 255, 0),  # Left ear - Yellow
    (170, 255, 0),  # Right ear - Yellow
    (85, 255, 0),   # Left shoulder - Green
    (0, 255, 0),    # Right shoulder - Green
    (0, 255, 85),   # Left elbow - Cyan
    (0, 255, 170),  # Right elbow - Cyan
    (0, 255, 255),  # Left wrist - Cyan
    (0, 170, 255),  # Right wrist - Cyan
    (0, 85, 255),   # Left hip - Blue
    (0, 0, 255),    # Right hip - Blue
    (85, 0, 255),   # Left knee - Purple
    (170, 0, 255),  # Right knee - Purple
    (255, 0, 255),  # Left ankle - Magenta
    (255, 0, 170)   # Right ankle - Magenta
]

# Line colors for skeleton visualization
LINE_COLORS = [
    (255, 85, 0),   # Face
    (255, 85, 0),   # Face
    (255, 170, 0),  # Face
    (255, 170, 0),  # Face
    (0, 255, 85),   # Left arm
    (0, 255, 170),  # Left arm
    (0, 255, 85),   # Right arm
    (0, 255, 170),  # Right arm
    (85, 255, 0),   # Torso
    (0, 85, 255),   # Torso
    (0, 85, 255),   # Torso
    (0, 0, 255),    # Torso
    (170, 0, 255),  # Left leg
    (255, 0, 255),  # Left leg
    (170, 0, 255),  # Right leg
    (255, 0, 255)   # Right leg
]


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-DETR Pose Estimation Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    
    # Dataset parameters
    parser.add_argument('--coco_path', type=str, required=True, 
                        help='Path to COCO dataset')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Optional path to a single image for inference')
    
    # Visualization parameters
    parser.add_argument('--output_dir', default='./visualization',
                        help='Path to save visualization results')
    parser.add_argument('--num_vis_images', type=int, default=10,
                        help='Number of images to visualize')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for visualization')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization mode (disables evaluation)')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results as COCO-style JSON')
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    # Inference parameters
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for inference (for evaluation mode)')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers for data loading')
    
    # Model configuration - these will be loaded from checkpoint but can be overridden
    parser.add_argument('--clip_model_name', default='openai/clip-vit-large-patch14-336', type=str,
                        help='Name of CLIP model to use')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Size of the embeddings')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help='Intermediate size of feedforward layers')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout applied in transformer')
    parser.add_argument('--num_body_points', default=17, type=int,
                        help='Number of keypoints')
    parser.add_argument('--image_input_size', default=336, type=int,
                        help='Model input image size')
    
    return parser


def load_model(args, checkpoint_path):
    """
    Load the CLIP-DETR model from a checkpoint.
    
    Args:
        args: Arguments with model configuration
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Loaded model in evaluation mode
        criterion: Loss criterion for evaluation
        postprocessors: Post-processing modules
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict and args from checkpoint
    model_state_dict = checkpoint['model']
    checkpoint_args = checkpoint.get('args', None)
    
    # Update args with checkpoint args if available
    if checkpoint_args is not None:
        for k, v in vars(checkpoint_args).items():
            if k not in ['output_dir', 'resume', 'eval', 'start_epoch']:
                setattr(args, k, v)
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.to(args.device)
    model.eval()
    
    # Move criterion to device
    criterion.to(args.device)
    
    return model, criterion, postprocessors


def visualize_keypoints(image, keypoints, scores, threshold=0.5):
    """
    Visualize detected keypoints on an image.
    
    Args:
        image: PIL image
        keypoints: Tensor of keypoints [N, num_keypoints*3] in xyxyvv format
        scores: Tensor of detection scores [N]
        threshold: Score threshold for visualization
        
    Returns:
        PIL image with keypoints visualization
    """
    # Convert to PIL image if it's a tensor or ndarray
    if not isinstance(image, Image.Image):
        if isinstance(image, torch.Tensor):
            # Convert from tensor [3, H, W] with normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
    
    # Create a copy for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    num_keypoints = len(COCO_KEYPOINT_NAMES)
    
    # For each valid detection
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        
        kpts = keypoints[i]
        
        # Draw keypoints
        for k in range(num_keypoints):
            x = kpts[k * 3]
            y = kpts[k * 3 + 1]
            v = kpts[k * 3 + 2]
            
            # Only draw visible keypoints
            if v > 0.1:
                # Draw a circle for each keypoint
                radius = 3
                color = KEYPOINT_COLORS[k]
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        
        # Draw skeleton lines
        for j, connection in enumerate(COCO_SKELETON):
            kp1_idx, kp2_idx = connection
            x1 = kpts[kp1_idx * 3]
            y1 = kpts[kp1_idx * 3 + 1]
            v1 = kpts[kp1_idx * 3 + 2]
            
            x2 = kpts[kp2_idx * 3]
            y2 = kpts[kp2_idx * 3 + 1]
            v2 = kpts[kp2_idx * 3 + 2]
            
            # Only draw lines for visible keypoint pairs
            if v1 > 0.1 and v2 > 0.1:
                draw.line([x1, y1, x2, y2], fill=LINE_COLORS[j], width=2)
        
        # Add confidence score
        draw.text((10, 10 + i * 20), f"Score: {scores[i]:.2f}", fill=(255, 255, 255))
    
    return draw_image

def matplotlib_visualize_keypoints(image, keypoints, scores, threshold=0.5, figsize=(12, 12)):
    """
    Visualize detected keypoints using matplotlib for better quality.
    """
    # Convert to numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        # Move to CPU first if on CUDA
        if image.is_cuda:
            image = image.cpu()
        # Convert from tensor [3, H, W] with normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    
    # Convert to numpy if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    # Make sure keypoints and scores are on CPU if they're on CUDA
    if isinstance(keypoints, torch.Tensor) and keypoints.is_cuda:
        keypoints = keypoints.cpu()
    if isinstance(scores, torch.Tensor) and scores.is_cuda:
        scores = scores.cpu()
    
    num_keypoints = len(COCO_KEYPOINT_NAMES)
    
    # For each valid detection
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        
        kpts = keypoints[i]
        
        # Draw keypoints
        for k in range(num_keypoints):
            x = kpts[k * 3].item()  # Convert to Python scalar
            y = kpts[k * 3 + 1].item()  # Convert to Python scalar
            v = kpts[k * 3 + 2].item()  # Convert to Python scalar
            
            # Only draw visible keypoints
            if v > 0.1:
                # Convert RGB to matplotlib format
                r, g, b = KEYPOINT_COLORS[k]
                color = (r/255, g/255, b/255)
                
                # Draw keypoint
                ax.scatter(x, y, s=50, c=[color], marker='o')
        
        # Draw skeleton lines
        for j, connection in enumerate(COCO_SKELETON):
            kp1_idx, kp2_idx = connection
            x1 = kpts[kp1_idx * 3].item()  # Convert to Python scalar
            y1 = kpts[kp1_idx * 3 + 1].item()  # Convert to Python scalar
            v1 = kpts[kp1_idx * 3 + 2].item()  # Convert to Python scalar
            
            x2 = kpts[kp2_idx * 3].item()  # Convert to Python scalar
            y2 = kpts[kp2_idx * 3 + 1].item()  # Convert to Python scalar
            v2 = kpts[kp2_idx * 3 + 2].item()  # Convert to Python scalar
            
            # Only draw lines for visible keypoint pairs
            if v1 > 0.1 and v2 > 0.1:
                r, g, b = LINE_COLORS[j]
                color = (r/255, g/255, b/255)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)
        
        # Add score text
        ax.text(10, 20 + i * 20, f"Score: {scores[i].item():.2f}", 
                color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
    
    # Remove axis ticks and labels
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax

# 이미지 랜덤 선택 부분을 전체 이미지를 순차적으로 처리하도록 변경
# (기존 random.sample 사용 부분 제거)

@torch.no_grad()
def run_inference(model, criterion, postprocessors, dataset, args):
    """
    Run inference on a dataset and visualize results.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataLoader for inference
    data_loader = DataLoader(
        dataset, 
        batch_size=1,  # For visualization, batch size should be 1
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    
    # 전체 이미지 중 처리할 이미지 수 (제한하고 싶다면)
    max_images = args.num_vis_images
    processed_count = 0
    
    # Process each image sequentially
    for i, (samples, targets) in enumerate(data_loader):
        if processed_count >= max_images:
            break
            
        # Move to device
        samples = samples.to(args.device)
        targets = [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in t.items()} for t in targets]
        
        # Run model
        outputs = model(samples)
        
        # Post-process results
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['keypoints'](outputs, orig_target_sizes)
        
        # Process each image in the batch
        for j, (target, result) in enumerate(zip(targets, results)):
            img_id = target['image_id'].item()
            
            # Get image path and load original image
            img_info = dataset.img_dict[img_id]
            img_path = os.path.join(dataset.root, img_info['file_name'])
            orig_img = Image.open(img_path).convert('RGB')
            
            # Get detection results
            scores = result['scores']
            labels = result['labels']
            keypoints = result['keypoints']
            
            # 점수에 따라 정렬하고 상위 10개만 유지
            if len(scores) > 0:
                # 내림차순으로 정렬할 인덱스 구하기
                sorted_indices = torch.argsort(scores, descending=True)
                
                # 상위 10개(또는 전체 길이 중 작은 쪽)만 유지
                max_detections = min(3, len(scores))
                sorted_indices = sorted_indices[:max_detections]
                
                # 정렬된 인덱스를 사용하여 결과 업데이트
                scores = scores[sorted_indices]
                labels = labels[sorted_indices]
                keypoints = keypoints[sorted_indices]
            
            # Filter by threshold
            mask = scores > args.threshold
            scores = scores[mask]
            labels = labels[mask]
            keypoints = keypoints[mask]
            
            # Skip if no detections
            if len(scores) == 0:
                print(f"No detections above threshold for image {img_id}")
                continue
            
            # Visualize results
            if args.visualize:
                # Use matplotlib for better visualization
                fig, ax = matplotlib_visualize_keypoints(
                    orig_img, keypoints, scores, args.threshold
                )
                
                # Add image ID and file name
                plt.title(f"Image ID: {img_id}, File: {img_info['file_name']}")
                
                # Save figure
                output_path = output_dir / f"vis_{img_id}.png"
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                print(f"Saved visualization to {output_path}")
            
            # 저장 코드는 그대로 유지...
            
            processed_count += 1
            if processed_count >= max_images:
                break
    
    print(f"Processed {processed_count} images.")


def inference_on_single_image(model, criterion, postprocessors, image_path, args):
    """
    Run inference on a single image file.
    
    Args:
        model: CLIP-DETR model
        criterion: Loss criterion (not used in single image mode)
        postprocessors: Post-processing modules
        image_path: Path to input image
        args: Arguments with inference configuration
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Get image dimensions
    w, h = image.size
    
    # Preprocess image (similar to dataset transforms)
    transform = T.Compose([
        T.Resize(336),  # Resize to match model input
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(args.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Post-process results
    target_size = torch.tensor([[h, w]]).to(args.device)
    results = postprocessors['keypoints'](outputs, target_size)[0]
    
    # Get detection results
    scores = results['scores']
    labels = results['labels']
    keypoints = results['keypoints']
    
    # Filter by threshold
    mask = scores > args.threshold
    scores = scores[mask]
    labels = labels[mask]
    keypoints = keypoints[mask]
    
    # Skip if no detections
    if len(scores) == 0:
        print(f"No detections above threshold for image {image_path}")
        return
    
    # Visualize results
    if args.visualize:
        # Use matplotlib for better visualization
        fig, ax = matplotlib_visualize_keypoints(
            image, keypoints, scores, args.threshold
        )
        
        # Add image file name
        plt.title(f"File: {os.path.basename(image_path)}")
        
        # Save figure
        output_path = output_dir / f"vis_{os.path.basename(image_path)}"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"Saved visualization to {output_path}")


@torch.no_grad()
def run_evaluation(model, criterion, postprocessors, data_loader, device, output_dir=None):
    """
    Run full evaluation on the COCO validation set.
    
    Args:
        model: Model to evaluate
        criterion: Loss criterion
        postprocessors: Post-processing modules
        data_loader: Evaluation data loader
        device: Device to evaluate on
        output_dir: Directory to save results
    """
    print("Running evaluation on COCO validation set...")
    stats, evaluator = evaluate(model, criterion, postprocessors, data_loader, device, output_dir)
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"AP: {stats['coco_eval_keypoints'][0]:.4f}")
    print(f"AP50: {stats['coco_eval_keypoints'][1]:.4f}")
    print(f"AP75: {stats['coco_eval_keypoints'][2]:.4f}")
    
    # Save evaluation results
    if output_dir:
        output_path = Path(output_dir) / "eval_results.txt"
        with open(output_path, 'w') as f:
            f.write("Evaluation Results:\n")
            f.write(f"AP: {stats['coco_eval_keypoints'][0]:.4f}\n")
            f.write(f"AP50: {stats['coco_eval_keypoints'][1]:.4f}\n")
            f.write(f"AP75: {stats['coco_eval_keypoints'][2]:.4f}\n")
            
            # Write all metrics
            f.write("\nAll Metrics:\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        
        print(f"Evaluation results saved to {output_path}")
    
    return stats, evaluator


def main(args):
    """
    Main function for inference, visualization, and evaluation.
    """
    # Set device
    device = torch.device(args.device)
    
    # Initialize distributed mode if available
    utils.init_distributed_mode(args)
    
    # Load model from checkpoint
    model, criterion, postprocessors = load_model(args, args.checkpoint)
    
    # Single image mode
    if args.image_path is not None:
        inference_on_single_image(model, criterion, postprocessors, args.image_path, args)
        return
    
    # Dataset mode
    print("Building dataset...")
    dataset = build_dataset(image_set='val', args=args)
    print(f"Dataset size: {len(dataset)}")
    
    if args.visualize:
        # Visualization mode
        print("Running visualization...")
        run_inference(model, criterion, postprocessors, dataset, args)
    else:
        # Evaluation mode
        print("Running evaluation...")
        
        # Create evaluation data loader
        sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn
        )
        
        # Run evaluation
        run_evaluation(model, criterion, postprocessors, data_loader, device, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-DETR Pose Estimation Inference', 
                                   parents=[get_args_parser()])
    args = parser.parse_args()
    
    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)