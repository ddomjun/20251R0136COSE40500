# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .CocoPoseDataset import build as build_coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
