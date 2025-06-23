import torch
import torch.nn as nn
import torch.nn.functional as F



# ===============================[ DeformConv2D ]===============================

class DeformConv2D(nn.Module): #Separate file

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = 0,
        dilation:     int = 1,
        groups:       int = 1,
        bias:         bool = True
    ):
        super(DeformConv2D, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Standard convolution to learn offsets
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,  # 2 for 'x' and 'y' offsets
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            bias = True
        )
        
        # Main convolution (standard, but will be used with offsets)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size = kernel_size, 
            stride = 1,  # No direct stride application here
            padding = 0,  # Padding managed by the offsets
            dilation = 1, 
            groups = groups, 
            bias = bias
        )
        
        self.init_weights()

    def init_weights(
        self
    ):

        nn.init.kaiming_uniform_(self.offset_conv.weight, a=1)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor
    ):

        # Offsets computation
        offsets = self.offset_conv(x)  # Shape: (B, 2 * K * K, H, W)
        B, C, H, W = x.size()

        # Generate a regular grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype = torch.float32, device=x.device),
            torch.arange(0, W, dtype = torch.float32, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Add offsets to the grid
        offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, 2 * K * K)
        offsets = offsets.view(B, H, W, -1, 2)  # (B, H, W, K * K, 2)
        grid = grid.unsqueeze(3) + offsets  # Apply offsets (B, H, W, K * K, 2)

        # Normalize the grid for F.grid_sample
        grid = (grid / torch.tensor([W - 1, H - 1], device=x.device) * 2) - 1  # Normalization [-1, 1]
        grid = grid.view(B, H, W, -1, 2)  # (B, H, W, K * K, 2)

        # Reshape to fit grid_sample
        grid = grid[:, :, :, 0, :]  # Take the first position (H_out = H_in, W_out = W_in)

        # Apply grid_sample
        x = F.grid_sample(
            x,
            grid,  # (B, H, W, 2)
            mode = 'bilinear',
            padding_mode = 'zeros',
            align_corners = True
        )

        return x