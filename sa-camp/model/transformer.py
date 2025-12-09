"""
Transformer components for enhancing CAMP with global message passing capabilities.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for global message passing between atoms.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of the input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Check if hidden dimension is divisible by number of heads
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Hidden dimension {hidden_dim} must be divisible by number of heads {num_heads}"
            )
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for query, key, value
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output features [batch_size, seq_len, hidden_dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Linear projections and reshape to [batch_size, num_heads, seq_len, head_dim]
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        out = self.output(out)
        
        return out
    
    
class FeedForward(nn.Module):
    """
    Feedforward network in transformer layer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of input and output
            ffn_dim: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output features [batch_size, seq_len, hidden_dim]
        """
        # First linear layer with GELU activation
        x = F.gelu(self.linear1(x))
        
        # Dropout
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x


class TransformerLayer(nn.Module):
    """
    A single transformer layer with multi-head attention and feedforward network.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_dim: int = 2048,
    ):
        """
        Args:
            hidden_dim: Dimension of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_dim: Dimension of feedforward network hidden layer
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feedforward network
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output features [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
        """
        # Handle unbatched input (add batch dimension if needed)
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)  # [1, seq_len, hidden_dim]
            
        # Attention block with residual connection and layer normalization
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.attention(x, mask))
        
        # Feedforward block with residual connection and layer normalization
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)  # [seq_len, hidden_dim]
            
        return x