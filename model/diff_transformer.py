"""
Differential Transformer components for enhancing CAMP with global message passing capabilities.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DifferentialMultiHeadAttention(nn.Module):
    """
    Differential Multi-head attention mechanism for global message passing between atoms.
    Implements the DIFF Transformer attention mechanism that reduces attention noise.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        layer_idx: int = 0,
    ):
        """
        Args:
            hidden_dim: Dimension of the input features
            num_heads: Number of attention heads
            dropout: Dropout probability
            layer_idx: Layer index for lambda_init calculation (0-indexed)
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
        
        # For differential attention, we need d_k = d_h / 2
        # So each Q/K group has dimension hidden_dim // 2
        self.diff_head_dim = self.head_dim // 2
        
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Head dimension {self.head_dim} must be even for differential attention"
            )
        
        # Linear projections for query (split into Q1 and Q2), key (split into K1 and K2), value
        # Q and K will produce 2 * hidden_dim outputs (for Q1, Q2 and K1, K2)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)  # Produces Q1 and Q2
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)  # Produces K1 and K2
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor (using d_k = d_h / 2)
        self.scale = math.sqrt(self.diff_head_dim)
        
        # Lambda initialization based on layer index
        # λ_init = 0.8 - 0.6 × exp(-0.3 × (ℓ-1))
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        
        # Learnable parameters for lambda reparameterization
        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        self.lambda_q1 = nn.Parameter(torch.randn(self.diff_head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.diff_head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.diff_head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.diff_head_dim) * 0.1)
        
        # GroupNorm: LayerNorm for each head independently
        # We apply this to each head's output before concatenation
        self.head_norms = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])
        
    def compute_lambda(self) -> Tensor:
        """
        Compute the learnable lambda parameter using reparameterization:
        λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        
        Returns:
            Lambda scalar value
        """
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        return lambda_1 - lambda_2 + self.lambda_init
        
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
        
        # Linear projections
        # For Q and K, we split into two groups: Q1, Q2 and K1, K2
        q = self.q_linear(x)  # [batch_size, seq_len, hidden_dim]
        k = self.k_linear(x)  # [batch_size, seq_len, hidden_dim]
        v = self.v_linear(x)  # [batch_size, seq_len, hidden_dim]
        
        # Split Q and K into two groups
        # Q: [batch_size, seq_len, hidden_dim] -> [Q1, Q2] each [batch_size, seq_len, hidden_dim/2]
        q1, q2 = q.chunk(2, dim=-1)  # Each: [batch_size, seq_len, hidden_dim/2]
        k1, k2 = k.chunk(2, dim=-1)  # Each: [batch_size, seq_len, hidden_dim/2]
        
        # Reshape for multi-head attention
        # Q1, Q2: [batch_size, seq_len, hidden_dim/2] -> [batch_size, num_heads, seq_len, diff_head_dim]
        q1 = q1.view(batch_size, seq_len, self.num_heads, self.diff_head_dim).transpose(1, 2)
        q2 = q2.view(batch_size, seq_len, self.num_heads, self.diff_head_dim).transpose(1, 2)
        k1 = k1.view(batch_size, seq_len, self.num_heads, self.diff_head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, self.num_heads, self.diff_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores for both Q-K pairs
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / self.scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores1 = scores1.masked_fill(mask == 0, -1e9)
            scores2 = scores2.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_weights2 = F.softmax(scores2, dim=-1)
        
        # Apply dropout
        attn_weights1 = self.dropout(attn_weights1)
        attn_weights2 = self.dropout(attn_weights2)
        
        # Compute lambda
        lambda_param = self.compute_lambda()
        
        # Differential attention: (softmax(Q1K1^T) - λ·softmax(Q2K2^T))V
        diff_attn_weights = attn_weights1 - lambda_param * attn_weights2
        
        # Apply differential attention weights to values
        out = torch.matmul(diff_attn_weights, v)
        # out: [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose to [batch_size, seq_len, num_heads, head_dim]
        out = out.transpose(1, 2).contiguous()
        
        # Apply GroupNorm: LayerNorm to each head independently
        # Split heads and apply normalization
        out_list = []
        for head_idx in range(self.num_heads):
            head_out = out[:, :, head_idx, :]  # [batch_size, seq_len, head_dim]
            # Apply LayerNorm with scaling factor (1 - λ_init)
            head_out = self.head_norms[head_idx](head_out)
            head_out = head_out * (1.0 - self.lambda_init)
            out_list.append(head_out)
        
        # Concatenate normalized heads
        # Stack: [batch_size, seq_len, num_heads, head_dim]
        out = torch.stack(out_list, dim=2)
        
        # Reshape and combine heads: [batch_size, seq_len, hidden_dim]
        out = out.view(batch_size, seq_len, self.hidden_dim)
        
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


class DifferentialTransformerLayer(nn.Module):
    """
    A single transformer layer with differential multi-head attention and feedforward network.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_dim: int = 2048,
        layer_idx: int = 0,
    ):
        """
        Args:
            hidden_dim: Dimension of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_dim: Dimension of feedforward network hidden layer
            layer_idx: Layer index for lambda_init calculation
        """
        super().__init__()
        
        # Differential multi-head attention
        self.attention = DifferentialMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_idx=layer_idx,
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


# Keep backward compatibility by aliasing to the differential versions
TransformerLayer = DifferentialTransformerLayer
MultiHeadAttention = DifferentialMultiHeadAttention