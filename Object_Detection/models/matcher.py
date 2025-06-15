"""
Hungarian matcher for bipartite matching between predictions and ground truth.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Performs Hungarian bipartite matching between predictions and ground truth targets.
    This matches each prediction to at most one ground truth and vice versa.
    
    The matching is done by solving a minimum-cost bipartite matching problem using
    the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """
        Initialize the matcher with cost coefficients.
        
        Args:
            cost_class: Coefficient for classification cost
            cost_bbox: Coefficient for L1 bounding box regression cost
            cost_giou: Coefficient for generalized IoU cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "at least one cost should be non-zero"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching between outputs and targets.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts with 'labels' and 'boxes'
            [
                {  # 첫 번째 이미지의 타겟 정보
                    'boxes': tensor([[x1_min, y1_min, x1_max, y1_max],   # 첫 번째 객체
                                    [x2_min, y2_min, x2_max, y2_max],   # 두 번째 객체
                                    ...]),                              # 형태: [이미지1의 객체 수, 4]
                    'labels': tensor([class_id1, class_id2, ...]),       # 형태: [이미지1의 객체 수]
                    'image_id': tensor([image_id1])                      # 이미지 ID
                    # 기타 메타데이터...
                },
                
                {  # 두 번째 이미지의 타겟 정보
                    'boxes': tensor([[x1_min, y1_min, x1_max, y1_max],
                                    [x2_min, y2_min, x2_max, y2_max],
                                    [x3_min, y3_min, x3_max, y3_max],
                                    ...]),                              # 형태: [이미지2의 객체 수, 4] 
                    'labels': tensor([class_id1, class_id2, class_id3, ...]),  # 형태: [이미지2의 객체 수]
                    'image_id': tensor([image_id2])
                    # 기타 메타데이터...
                },
                
                # 배치 내 다른 이미지들...
            ]       
            
        Returns:
            List of tuples with indices of matched predictions and targets
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        # Compute classification cost as 1 - probability of the target class
        cost_class = -out_prob[:, tgt_ids] # [batch_size * num_queries, total_num_targets]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [batch_size * num_queries, total_num_targets]

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) # [batch_size * num_queries, total_num_targets]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # [batch_size * num_queries, total_num_targets]
        C = C.view(bs, num_queries, -1).cpu() # [batch_size, num_queries, total_num_targets]

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # [([23, 45], [0, 1]), ([23, 45, 17], [0, 1, 2]), (row_indices, col_indices)...] 
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # [([23, 45], [0, 1]), ([23, 45, 17], [0, 1, 2]), (row_indices, col_indices)...] 


def build_matcher(args):
    """
    Build Hungarian matcher from arguments.
    
    Args:
        args: Arguments containing matcher configuration
        
    Returns:
        matcher: Hungarian matcher
    """
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )
