import os
import json
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import List, Dict, Tuple, Any, Optional
from util.box_ops import box_xyxy_to_cxcywh
from pathlib import Path
"""
output: 
image: padded, resized
target = {
            'image_id': torch.tensor([img_id]),
            'boxes': 
            'labels': torch.empty((0,), dtype=torch.int64),
            'keypoints': 
            'area': torch.empty((0,), dtype=torch.float32),
            'iscrowd': torch.empty((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']]),
            'size': torch.as_tensor([img_info['height'], img_info['width']]) : 336*336( 변환 사이즈 )
        }
        boxes: [N, 4], [cx, cy, w, h], padding, resize, normalize(0-1)
        keypoints: [N, 17*3], xyxyvv, 각 bbox 기준 0-1 정규화
        
"""
class CocoPoseDataset(VisionDataset):
    """
    COCO 데이터셋 형식을 지원하지만 COCO API를 사용하지 않는 커스텀 데이터셋
    """
    def __init__(
        self, 
        img_folder: str, 
        ann_file: str, 
        transforms=None
    ):
        super(CocoPoseDataset, self).__init__(
            img_folder, transforms=transforms
        )
        # 주석 파일 로드
        with open(ann_file, 'r') as f:
            self.coco_json = json.load(f)
        
        # 이미지 ID 목록 및 카테고리 정보 추출
        self.img_dict = {img['id']: img for img in self.coco_json['images']}
        self.cats = {cat['id']: cat for cat in self.coco_json['categories']}
        
        # 이미지 ID별 주석 맵 생성
        self.ann_map = {}
        for ann in self.coco_json['annotations']:
            img_id = ann['image_id']
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)

        self.ids = []
        for img_id in sorted(self.img_dict.keys()):
            # 해당 이미지에 주석이 없으면 건너뛰기
            if img_id not in self.ann_map:
                continue
            
            # 해당 이미지의 모든 주석에서 키포인트 수 합산
            annotations = self.ann_map[img_id]
            num_keypoints = [obj.get('num_keypoints', 0) for obj in annotations]
            
            # 키포인트 합이 0이면 이미지 제외
            if sum(num_keypoints) == 0:
                continue
                
            self.ids.append(img_id)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        index에 해당하는 이미지와 주석 반환
        """
        img_id = self.ids[index]
        img_info = self.img_dict[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # 이미지 로드
        img = Image.open(img_path).convert('RGB')
        
        # 해당 이미지의 주석 가져오기
        annotations = self.ann_map.get(img_id, [])
        
        # 타겟 딕셔너리 생성
        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64),
            'keypoints': torch.empty((0, 17, 3), dtype=torch.float32),
            'area': torch.empty((0,), dtype=torch.float32),
            'iscrowd': torch.empty((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([img_info['height'], img_info['width']]),
            'size': torch.as_tensor([img_info['height'], img_info['width']])
        }
        
        # 주석이 있는 경우 처리
        if annotations:
            boxes = []
            labels = []
            areas = []
            iscrowds = []
            keypoints = []
            
            for ann in annotations:
                if ann['num_keypoints'] == 0:
                    continue
                if 'bbox' not in ann or not ann['bbox']:
                    continue
                
                # COCO 형식의 bbox는 [x, y, width, height]
                keypoints.append(ann['keypoints']) # 지금은 [17*3]으로 xyvxyv
                x, y, w, h = ann['bbox']
                # 변환 to [x1, y1, x2, y2]
                w = min(w, img_info['width'] - x)
                h = min(h, img_info['height'] - y)
                boxes.append([x, y, x + w, y + h]) # [x1, y1, x2, y2]
                labels.append(ann['category_id'])
                areas.append(ann['area'])
                iscrowds.append(ann.get('iscrowd', 0))
                
                
            
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32) # [N, 4] => [x1, y1, x2, y2]
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(iscrowds, dtype=torch.int64)
            target['keypoints'] = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3) # [N, 17, 3] xyv
        
        # 변환 적용
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self) -> int:
        return len(self.ids)


class SquarePadAndResize(object):
    """
    이미지를 정사각형으로 패딩한 후 지정된 크기로 리사이징
    키포인트를 바운딩 박스 기준으로 정규화 (패딩/리사이징 전 원본 좌표 기준)
    """
    def __init__(self, size):
        self.size = size  # 리사이징할 정사각형 크기
        if isinstance(size, int):
            self.size = (size, size)
    
    def __call__(self, image, target=None):
        w, h = image.size
        
        # 원본 좌표에서 키포인트를 바운딩 박스 기준으로 정규화 (패딩/리사이징 전)
        if "keypoints" in target and target["keypoints"].shape[0] > 0 and "boxes" in target and len(target["boxes"]) > 0:
            keypoints = target["keypoints"].clone()  # [N, 17, 3]
            boxes = target["boxes"].clone()  # [N, 4] [x1, y1, x2, y2]
            
            # 각 객체(person)별로 처리하여 키포인트를 바운딩 박스 기준으로 정규화
            for i in range(keypoints.shape[0]):
                # 바운딩 박스 좌표 추출
                x1, y1, x2, y2 = boxes[i]
                box_w = x2 - x1
                box_h = y2 - y1
                
                # 키포인트의 x, y 좌표를 해당 바운딩 박스 기준으로 정규화
                keypoints[i, :, 0] = (keypoints[i, :, 0] - x1) / box_w
                keypoints[i, :, 1] = (keypoints[i, :, 1] - y1) / box_h
                
                # 0-1 범위를 벗어나는 값 클리핑 (박스 밖의 키포인트 처리)
                keypoints[i, :, 0] = torch.clamp(keypoints[i, :, 0], 0.0, 1.0)
                keypoints[i, :, 1] = torch.clamp(keypoints[i, :, 1], 0.0, 1.0)
            
            target["keypoints"] = keypoints # [N, 17, 3] 0-1 범위로 정규화
        
        # 이제 이미지에 패딩 및 리사이징 적용
        # 긴 변 기준으로 정사각형 크기 계산
        max_size = max(w, h)
        
        # 패딩 계산 (이미지는 중앙에 위치하도록)
        left_pad = (max_size - w) // 2
        right_pad = max_size - w - left_pad
        top_pad = (max_size - h) // 2
        bottom_pad = max_size - h - top_pad
        
        # 패딩 적용 (여백은 흰색 또는 회색으로)
        padding = (left_pad, top_pad, right_pad, bottom_pad)
        padded_image = F.pad(image, padding, fill=(128, 128, 128))
        
        # 리사이징 비율 계산
        scale_w = self.size[1] / max_size
        scale_h = self.size[0] / max_size
        
        # 바운딩 박스 좌표만 패딩 및 리사이징 적용
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            # 패딩 적용에 따른 좌표 조정
            boxes[:, 0] += left_pad  # x1
            boxes[:, 2] += left_pad  # x2
            boxes[:, 1] += top_pad   # y1
            boxes[:, 3] += top_pad   # y3
            
            # 리사이징에 따른 좌표 조정
            boxes[:, 0] *= scale_w
            boxes[:, 2] *= scale_w
            boxes[:, 1] *= scale_h
            boxes[:, 3] *= scale_h
            
            target["boxes"] = boxes
        
        # 크기 정보 업데이트
        target["orig_size"] = torch.as_tensor([h, w])
        target["size"] = torch.as_tensor(self.size)
        
        # 리사이징 적용
        resized_image = F.resize(padded_image, self.size)
        
        return resized_image, target

class ToTensorAndNormalize(object):
    """
    이미지를 텐서로 변환하고 정규화
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        # PIL 이미지를 텐서로 변환
        image = F.to_tensor(image)
        # 정규화 적용
        image = F.normalize(image, mean=self.mean, std=self.std)
        
        # 바운딩 박스 좌표 변환 및 정규화
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            h, w = target["size"]
            
            # [x1, y1, x2, y2] -> [cx, cy, w, h] 형식으로 변환
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            
            # 바운딩 박스를 이미지 크기로 나누어 0-1 범위로 정규화
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            
            target["boxes"] = boxes
        
        # 키포인트 가시성 처리
        if target is not None and "keypoints" in target and target["keypoints"].shape[0] > 0:
            h, w = target["size"]
            keypoints = target["keypoints"].clone()  # [N, 17, 3]
            
            # 가시성 처리 (visibility=2인 경우 1로 변환)
            keypoints[:, :, 2][keypoints[:, :, 2] == 2] = 1
            
            # GroupPose 형식으로 변환 (xyzxyz -> xyxy...zz...)
            V = keypoints[:, :, 2]  # [N, 17]
            Z = keypoints[:, :, :2].reshape(keypoints.shape[0], -1)  # [N, 34]
            all_keypoints = torch.cat([Z, V], dim=1)
            
            target["keypoints"] = all_keypoints
        
        return image, target


def make_clip_transforms(size=336):
    """
    CLIP 모델을 위한 변환 함수
    """
    return Compose([
        SquarePadAndResize(size),
        ToTensorAndNormalize(),
    ])


class Compose(object):
    """
    여러 변환을 순차적으로 적용
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'person_keypoints'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        #"val": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoPoseDataset(img_folder, ann_file, transforms=make_clip_transforms(size=336))
    return dataset