import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class QFormer(nn.Module):
    """
    Modified Q-Former to handle variable number of queries per image in batch.
    Simplified version without memory key padding mask.
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        
        # Single transformer decoder layer for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=normalize_before,
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.d_model = d_model
        
    def forward(self, memory, query_embed, padding_mask=None):
        """
        Forward pass for Q-Former with variable number of queries per image.
        
        Args:
            memory: Encoded image features from CLIP [batch_size, num_tokens, hidden_dim]
            query_embed: Query embeddings [batch_size, max_queries, hidden_dim]
            padding_mask: Mask for queries to handle variable lengths [batch_size, max_queries]
            
            
        Returns:
            hs: Updated query embeddings [batch_size, max_queries, hidden_dim]
            memory: Original memory
        """
        bs, max_queries = query_embed.shape[:2]
        
        # Prepare query embeddings
        # Convert from [batch_size, max_queries, hidden_dim] to [max_queries, batch_size, hidden_dim]
        query_embed = query_embed.permute(1, 0, 2)
        
        # Convert memory from [batch_size, num_tokens, hidden_dim] to [num_tokens, batch_size, hidden_dim]
        memory = memory.permute(1, 0, 2)
        
        # Apply the transformer decoder
        hs = self.decoder(
            query_embed,                      # tgt: [max_queries, batch_size, hidden_dim]
            memory,                           # memory: [num_tokens, batch_size, hidden_dim]
            tgt_key_padding_mask=padding_mask if padding_mask is not None else None,  # [batch_size, max_queries]
        )
        
        # Convert back to [batch_size, max_queries, hidden_dim]
        hs = hs.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        
        return hs, memory