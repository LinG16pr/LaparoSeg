import torch
import torch.nn as nn



# ===============================[ MemoryAttentionBlock ]===============================

class MemoryAttentionBlock(nn.Module):

    def __init__(
        self,
        dim:           int,
        num_heads:     int,
        memory_frames: int = 4
    ):
        super(MemoryAttentionBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, batch_first = True)
        self.cross_attn = nn.MultiheadAttention(embed_dim = dim, num_heads = num_heads, batch_first = True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.memory_frames = memory_frames

    def forward(
        self,
        x:      torch.Tensor,
        memory: torch.Tensor
    ):
        """
        x: Tensor of shape (batch_size, N, dim) - Current frame
        memory: Tensor of shape (batch_size, memory_frames, N, dim) - Past frames and results
        """

        # Self-attention
        x_res = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x + x_res

        # Cross-attention with memory
        x_res = x
        x = self.norm2(x)
        memory_flat = memory.view(memory.size(0), -1, memory.size(-1))  # (batch_size, memory_frames * N, dim)
        x, _ = self.cross_attn(x, memory_flat, memory_flat)
        x = x + x_res

        # MLP
        x_res = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = x + x_res

        return x



# ===============================[ MemoryAttention ]===============================

class MemoryAttention(nn.Module):

    def __init__(
        self,
        dim:           int,
        num_heads:     int,
        memory_frames: int = 4,
        num_blocks:    int = 4
    ):
        super(MemoryAttention, self).__init__()

        self.blocks = nn.ModuleList([
            MemoryAttentionBlock(dim, num_heads, memory_frames) for _ in range(num_blocks)
        ])

    def forward(
        self,
        x:      torch.Tensor,
        memory: torch.Tensor
    ):
        """
        x: Tensor of shape (batch_size, N, dim) - Current frame
        memory: Tensor of shape (batch_size, memory_frames, N, dim) - Past frames and results
        """

        for block in self.blocks:
            x = block(x, memory)
            
        return x