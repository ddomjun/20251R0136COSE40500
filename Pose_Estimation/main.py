"""
Main training script for CLIP-DETR object detection model.

This file could be simplified by using more standard PyTorch training utilities.
"""
import wandb
import argparse
from datetime import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models.clip_detr import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-DETR Pose Estimation', add_help=False)
    

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # Model parameters
    ############### [build_model(args)] for clip_detr ####################
    parser.add_argument('--clip_model_name', default='openai/clip-vit-large-patch14-336', type=str,
                        help='Name of CLIP model to use')
    # parser.add_argument('--num_classes', default=2, type=int,
    #                     help="Number of classes")
    # parser.add_argument('--num_queries', default=30, type=int,
    #                     help="Number of query slots")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)") 
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions") 
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dataset_file', default='coco') # dadtaset name to decide class num
    
    ############### [build_model(args)] for build_matcher ####################
    # parser.add_argument('--set_cost_class', default=2, type=float,
    #                     help="Class coefficient in the matching cost")
    # parser.add_argument('--set_cost_keypoints', default=10, type=float,
    #                     help="L1 keypoint coefficient in the matching cost")
    # parser.add_argument('--set_cost_oks', default=4, type=float,
    #                     help="oks keypoints coefficient in the matching cost")
    
    ############### [build_model(args)] for weight dictionary parameter ####################
    # parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--visibility_loss_coef', default=0.1, type=float)
    parser.add_argument('--keypoints_loss_coef', default=2, type=float)
    parser.add_argument('--oks_loss_coef', default=3, type=float)
    
    ############### [build_model(args)] for SetCriterion ####################
    # parser.add_argument('--eos_coef', default=0.1, type=float,
    #                     help="Relative classification weight of the no-object class")
    
    ############### for pose estimation ####################
    parser.add_argument('--num_body_points', default=17, type=float)
    
    ############### train / eval ####################
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--grad_acc', default=1, type=int)
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=300, type=int)
    
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--image_input_size', default=336, type=int) # clip model input image size
    

    ############### dataset ####################
    parser.add_argument('--coco_path', type=str, default='./data/coco')
    parser.add_argument('--num_workers', default=2, type=int)
    
    ################## etc #######################
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser


def main(args):
    # Initialize distributed mode if available
    utils.init_distributed_mode(args)
    
    # Check device
    device = torch.device(args.device)
    
    if utils.is_main_process():  # 분산 학습에서 중복 로깅 방지
        wandb.init(
            project="VLM_MT",  # 프로젝트 이름
            config=vars(args),  # 하이퍼파라미터 로깅
            name=f"Pose_Only-{datetime.now().strftime('%Y%m%d-%H%M%S')}"  # 실행 이름
        )

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build model, criterion and postprocessors
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # Prepare for distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    ############## added from origin [mj] ###################
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # Set parameters that need different learning rates
    param_dicts = [
        # CLIP model parameters are already frozen
        {"params": [p for n, p in model_without_ddp.named_parameters()
               if "clip_projector.projection" in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters()
               if "qformer" in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters()
               if "query_embed" in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters()
               if ("pe_layer" in n or "keypoint_share_projection" in n or "keypoint_split_projection" in n) and p.requires_grad]},
        # Catch any remaining parameters that might not be covered
        {"params": [p for n, p in model_without_ddp.named_parameters()
                if not any(x in n for x in ["clip_projector.projection", "qformer", "query_embed", "pe_layer", "keypoint_share_projection", "keypoint_split_projection"]) 
                and p.requires_grad]}
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                 weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Create dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    # Create samplers for distributed training
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # Create batch samplers
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    # Create dataloaders
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers, collate_fn=utils.collate_fn,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=2)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 num_workers=args.num_workers, collate_fn=utils.collate_fn,
                                pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    
    # Output directory
    output_dir = Path(args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        ###### 아래 한줄 추가 [mj] ######
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        
    # Evaluation mode
    if args.eval:
        test_stats, evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(evaluator.evaluate(), output_dir / "eval.pth")  # 평가 결과 저장 방식 변경
        return

    
    # Training loop
    print("Start training")
    start_time = time.time()
    grad_acc = args.grad_acc
    global_step = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats, global_step = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, grad_acc, global_step)
            
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        
        # Evaluate on validation set
        test_stats, evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )
        
        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        # Save stats
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # for evaluation logs
            if evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                filenames = ['latest.pth']
                filenames.append(f'{epoch:03}.pth')
                for name in filenames:
                    torch.save(evaluator.evaluate(), output_dir / "eval" / name)
        ##################
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
        if utils.is_main_process():
            eval_results = test_stats.get('coco_eval_keypoints', [0, 0, 0, 0])
            wandb.log({
                "val/epoch": epoch,
                "val/loss": test_stats['loss'],
                "val/accuracy_50": eval_results[0],  # accuracy_50
                "val/accuracy_70": eval_results[1], # accuracy_70
                "val/accuracy_90": eval_results[2],    # accuracy_90
                "val/avg_OKS": eval_results[3]         # avg_oks
            })
    ########################
    if utils.is_main_process():
        wandb.finish()

    # Final logging
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    wandb.login()
    parser = argparse.ArgumentParser('CLIP-DETR training and evaluation script', 
                                    parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)
