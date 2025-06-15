"""
Q-Former module for CLIP-DETR with a single transformer decoder layer.

This file can be simplified by using torch's built-in TransformerDecoder components
rather than reimplementing the transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class QFormer(nn.Module):
    """
    Q-Former for CLIP-DETR with a single transformer decoder layer. 
    Takes image tokens from the CLIP model and uses learnable queries.
    
    This implementation uses PyTorch's built-in transformer components.
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
        """
        Initialize Q-Former with a single decoder layer.
        
        Args:
            d_model: Feature dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            normalize_before: Whether to normalize before or after sublayers
        """
        super().__init__()

        # Create a standard PyTorch transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=normalize_before
        )
        
        # Create a decoder with a single layer
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Layer normalization if needed
        self.norm = nn.LayerNorm(d_model) if normalize_before else None

        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        Initialize parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed=None):
        """
        Forward pass of Q-Former.
        
        Args:
            src: CLIP tokens [batch_size, num_tokens, hidden_dim] where HW is the number of tokens and C is feature dimension
            mask: Attention mask [batch_size, num_tokens] where True means to mask
            query_embed: Learnable query embeddings [num_queries, hidden_dim]
            pos_embed: Position embeddings for src tokens [B, H*W, C] or None
            
        Returns:
            Tuple of:
                - hs: Decoder output [1, B, num_queries, C] (batch of 1 layer)
                - memory: Processed encoder output [B, HW, C]
        """
        # Prepare input for decoder (using CLIP tokens as memory)
        bs, _, _ = src.shape
        num_queries = query_embed.weight.shape[0]
        
        # Generate queries from query embeddings
        t_input_query = query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, B, hidden_dim]
        
        # Process memory (CLIP tokens) - transpose for decoder
        memory = src.permute(1, 0, 2)  # [num_tokens, batch_size, hidden_dim]
        
        # Create position embeddings for queries (built-in to the query_embed)
        # query_pos = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, B, C]
        
        # Process position embeddings if available
        # if pos_embed is not None:
        #     pos_embed = pos_embed.permute(1, 0, 2)  # [HW, B, C]
        # else:
        #     pos_embed = torch.zeros_like(memory)
        
        # Create memory mask for the decoder (different format than input mask)
        if mask is not None:
            memory_key_padding_mask = mask # [B, HW]
        else:
            memory_key_padding_mask = None
        
        # Forward through the transformer decoder
        # Use PyTorch's implementation which expects different format
        output = self.decoder(
            t_input_query, memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=False
        )
        
        # Apply normalization if needed
        if self.norm is not None:
            output = self.norm(output)
        
        # Add batch dimension to match expected output shape [1, B, num_queries, C]
        # output = output.unsqueeze(0)
        
        # Return decoder output and memory
        return output.transpose(0, 1), src # ouitput: [B, num_queries, hidden_dim],  [B, num_tokens, hidden_dim]
