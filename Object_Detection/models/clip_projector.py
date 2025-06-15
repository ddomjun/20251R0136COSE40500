import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from typing import Tuple

class ClipProjector(nn.Module):
    """
    CLIP token projector module - CLIPVisionModel을 사용하여
    이미지의 마지막 hidden state(패치 토큰들)를 추출하고,
    이를 transformer의 임베딩 차원으로 프로젝션하는 모듈.
    
    Hugging Face의 CLIPVisionModel을 사용하며, output_hidden_states=True로 설정되어
    각 레이어의 hidden state를 반환합니다.
    """
    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14-336", projection_dim: int = 256): # , layer: int = -1
        """
        Args:
            clip_model_name: Hugging Face에서 불러올 CLIP 모델 이름
            projection_dim: 프로젝션 후 transformer의 임베딩 차원 (d_model)
            layer: 사용할 transformer 레이어의 hidden state (-1이면 마지막 레이어)
        """
        super().__init__()
        # CLIPVisionModel 로드 (output_hidden_states=True 설정)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name) # utput_hidden_states=True
        
        # CLIP 파라미터 동결
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 모델의 hidden size를 가져옴 (보통 config.hidden_size)
        self.feature_dim = self.clip_model.config.hidden_size
        
        # CLIP feature 차원을 transformer 임베딩 차원으로 프로젝션하는 레이어
        self.projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim),  # 첫 번째 선형 계층
                                        nn.ReLU(),                                     # ReLU 활성화 함수
                                        nn.Linear(self.feature_dim, projection_dim)     # 두 번째 선형 계층
                                        )
        # self.layer = layer

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CLIPVisionModel을 이용해 이미지의 token features(패치 토큰)를 추출하고 프로젝션합니다.
        
        Args:
            images: [batch_size, 3, H, W] 형태의 이미지 텐서
            
        Returns:
            projected_features: [batch_size, num_tokens, projection_dim] 형태의 프로젝션된 토큰들
            mask: 토큰에 대한 어텐션 마스크 (False이면 attend)
        """
        device = images.device
        self.clip_model.to(device)
        
        with torch.no_grad():
            outputs = self.clip_model(images, interpolate_pos_encoding=True) # interpolate cnrk [MJ]
            token_features = outputs.last_hidden_state
            # 일반적으로 첫번째 토큰은 [CLS] 토큰이므로, 패치 토큰만 사용하려면 이를 제외합니다.
            token_features = token_features[:, 1:, :]  # [batch_size, num_tokens, hidden_size]
            
            batch_size, num_tokens, _ = token_features.shape
            # 어텐션 마스크 생성 (False이면 마스킹하지 않음)
            mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
        
        # 프로젝션을 통해 transformer 임베딩 차원으로 변환
        projected_features = self.projection(token_features)
        return projected_features, mask