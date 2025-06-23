import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_convolution import DeformConv2D



# ===============================[ MSMCConv ]===============================

class MSMCConv(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int
    ):
        super(MSMCConv, self).__init__()

        # Standard depthwise convolutions with 3x3 kernel
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Standard depthwise convolutions with 5x5 kernel
        self.dwconv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 5, padding = 2, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Standard depthwise convolutions with 7x7 kernel
        self.dwconv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 7, padding = 3, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Pointwise convolution to combine the concatenated features
        self.pwconv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        # Apply the three parallel depthwise convolutions
        x1 = self.dwconv3x3(x)
        x2 = self.dwconv5x5(x)
        x3 = self.dwconv7x7(x)
        
        # Concatenate along the channel dimension
        x_concat = torch.cat([x1, x2, x3], dim = 1)
        
        # Apply the pointwise convolution
        out = self.pwconv(x_concat)
        
        return out
    


# ===============================[ DeformMSMCConv ]===============================

class DeformMSMCConv(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int
    ):
        super(DeformMSMCConv, self).__init__()

        # Deformable depthwise convolutions with 3x3 kernel
        self.dwconv3x3 = nn.Sequential(
            DeformConv2D(in_channels, in_channels, kernel_size = 3, padding = 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Deformable depthwise convolutions with 5x5 kernel
        self.dwconv5x5 = nn.Sequential(
            DeformConv2D(in_channels, in_channels, kernel_size = 5, padding = 2, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Deformable depthwise convolutions with 7x7 kernel
        self.dwconv7x7 = nn.Sequential(
            DeformConv2D(in_channels, in_channels, kernel_size = 7, padding = 3, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        # Pointwise convolution to combine the concatenated features
        self.pwconv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        # Apply the three parallel depthwise convolutions
        x1 = self.dwconv3x3(x)
        x2 = self.dwconv5x5(x)
        x3 = self.dwconv7x7(x)
        
        # Concatenate along the channel dimension
        x_concat = torch.cat([x1, x2, x3], dim = 1)
        
        # Apply the pointwise convolution
        out = self.pwconv(x_concat)
        
        return out



# ===============================[ BRA ]===============================
    
class BRA(nn.Module):
    
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        num_heads:  int = 8,
        k:          int = 5
    ):
        super(BRA, self).__init__()

        self.num_heads = num_heads
        self.k = k  # Number of regions for routing
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor
    ):

        # x shape: (batch_size, N, input_dim), where N = H * W
        B, N, C = x.shape
        Q = self.query_proj(x)  # Shape: (B, N, output_dim)
        K = self.key_proj(x)   # Shape: (B, N, output_dim)
        V = self.value_proj(x) # Shape: (B, N, output_dim)

        # Ensure regions can be divided evenly
        num_regions = self.k
        remainder = N % num_regions
        if remainder != 0:
            pad_size = num_regions - remainder
            K = F.pad(K, (0, 0, 0, pad_size))  # Pad along sequence dimension
            V = F.pad(V, (0, 0, 0, pad_size))  # Pad along sequence dimension
            Q = F.pad(Q, (0, 0, 0, pad_size))  # Pad along sequence dimension
            N += pad_size

        # Compute region size after padding
        region_size = N // num_regions

        # Reshape for region-to-region attention
        K_gathered = K.view(B, num_regions, region_size, -1)  # Shape: (B, k, region_size, C)
        A = torch.einsum("bqd,bgkd->bqgk", Q, K_gathered)  # Routing scores (B, N, k, region_size)
        A = F.softmax(A, dim=-1)  # Apply softmax

        # Aggregate values with routing scores
        V_gathered = V.view(B, num_regions, region_size, -1)
        output = torch.einsum("bqgk,bgkd->bqd", A, V_gathered)  # Shape: (B, N, output_dim)

        # Remove padding if added
        if remainder != 0:
            output = output[:, :N - pad_size]

        return self.output_proj(output)



# ===============================[ AttentionBlock ]===============================

class AttentionBlock(nn.Module):

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        num_heads:  int = 8,
        k:          int = 5
    ):
        super(AttentionBlock, self).__init__()

        self.dw_conv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.bra = BRA(input_dim, output_dim, num_heads=num_heads, k=k)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim)
        )
        self.layer_norm3 = nn.LayerNorm(output_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        # x shape: (batch_size, H, W, input_dim)
        shortcut = x
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, input_dim, H, W) for Conv2D
        x = self.dw_conv(x)        # Depthwise Conv
        x = x + shortcut.permute(0, 3, 1, 2)  # Residual connection
        x = x.permute(0, 2, 3, 1) # Back to (batch_size, H, W, input_dim)
        x = self.layer_norm1(x)

        # Flatten for BRA
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)  # (batch_size, H * W, input_dim)
        x = self.bra(x)          # Apply BRA
        x = self.layer_norm2(x)  # Normalize after BRA

        # Reshape back to 2D spatial dimensions
        x = x.view(B, H, W, -1)

        # First MLP
        shortcut2 = x
        x = x.view(B, H * W, -1)  # Flatten for MLP
        x = self.mlp1(x)
        x = x + shortcut2.view(B, H * W, -1)  # Residual connection
        x = x.view(B, H, W, -1)
        x = self.layer_norm3(x)  # Normalize after first MLP

        # Second MLP
        shortcut3 = x
        x = x.view(B, H * W, -1)  # Flatten for MLP
        x = self.mlp2(x)
        x = x + shortcut3.view(B, H * W, -1)  # Residual connection
        x = x.view(B, H, W, -1)

        return x
    


# ===============================[ EncoderBlock ]===============================
    
class EncoderBlock(nn.Module):

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        num_heads:  int  = 8,
        k:          int  = 5,
        downsample: bool = True,
        deformable: bool = True
    ):
        super(EncoderBlock, self).__init__()

        # Initial Conv3x3 layer
        self.conv3x3 = nn.Conv2d(input_dim, output_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace = True)

        # MSMCConv and AttentionBlock
        if deformable:
            self.msmc = DeformMSMCConv(output_dim, output_dim)
        else:
            self.msmc = MSMCConv(output_dim, output_dim)            
        self.attention_block = AttentionBlock(output_dim, output_dim, num_heads=num_heads, k=k)

        # Downsampling layer
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else None

    def forward(
        self,
        x: torch.Tensor
    ):

        # Initial Conv3x3
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)

        # Outputs from MSMCConv and AttentionBlock
        msmc_out = self.msmc(x)
        attn_out = self.attention_block(x.permute(0, 2, 3, 1))  # Change to (B, H, W, C)
        attn_out = attn_out.permute(0, 3, 1, 2)  # Back to (B, C, H, W)

        # Fusion of MSMCConv and AttentionBlock outputs
        fused_out = msmc_out + attn_out

        fusion = fused_out

        # Downsampling
        if self.downsample:
            fused_out = self.downsample(fused_out)

        return fused_out, fusion



# ===============================[ FrameEncoder ]===============================

class FrameEncoder(nn.Module):

    def __init__(
        self,
        input_dim:  int,
        num_blocks: int  = 5,
        base_dim:   int  = 64,
        num_heads:  int  = 8,
        k:          int  = 5,
        deformable: bool = True
    ):
        super(FrameEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        current_dim = input_dim

        for i in range(num_blocks):
            next_dim = base_dim * (2 ** i)
            downsample = i < num_blocks - 1 # Set downsample to False for the last block
            self.blocks.append(EncoderBlock(current_dim, next_dim, num_heads = num_heads, k = k, downsample = downsample, deformable = deformable))
            current_dim = next_dim

    def forward(
        self,
        x: torch.Tensor
    ):

        skips = []
        i = 1
        for block in self.blocks:
            x, fusion = block(x)
            if  i < len(self.blocks):
                skips.append(fusion)  # Store intermediate outputs if needed (for skip connections)
            i += 1

        return x, skips