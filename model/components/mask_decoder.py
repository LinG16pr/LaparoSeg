import torch
import torch.nn as nn

from .frame_encoder import MSMCConv, DeformMSMCConv



# ===============================[ DecoderBlock ]===============================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        deformable: bool = True
    ):
        super(DecoderBlock, self).__init__()

        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size = 2, stride = 2)

        # MSMCConv and AttentionBlock
        if deformable:
            self.msmc = DeformMSMCConv(output_dim, output_dim)
        else:
            self.msmc = MSMCConv(output_dim, output_dim)

        # Final Conv3x3 layer
        self.conv3x3 = nn.Conv2d(output_dim, output_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace = True)

    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor
    ):

        # Upsample the input
        x = self.upsample(x)

        # Add skip connection
        x = x + skip_connection

        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)

        out = self.msmc(x)

        return out



# ===============================[ MaskDecoder ]===============================

class MaskDecoder(nn.Module):

    def __init__(
        self,
        final_dim:   int,
        num_classes: int  = 2,
        num_blocks:  int  = 4,
        base_dim:    int  = 64,
        #num_heads:   int  = 8,
        #k:           int  = 5,
        deformable:  bool = True
    ):
        super(MaskDecoder, self).__init__()

        self.blocks = nn.ModuleList()
        current_dim = final_dim

        # Create decoder blocks in reverse order of the encoder
        for i in range(num_blocks - 1, -1, -1):
            next_dim = base_dim * (2 ** i)
            #self.blocks.append(DecoderBlock(current_dim, next_dim, num_heads = num_heads, k = k, deformable = deformable))
            self.blocks.append(DecoderBlock(current_dim, next_dim, deformable = deformable))
            current_dim = next_dim

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(base_dim, num_classes, kernel_size = 1) # Reduce channels to num_classes

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs
    ):

        # Pass through the decoder blocks
        for i, block in enumerate(self.blocks):
            # Use the corresponding skip connection from encoder outputs
            x = block(x, encoder_outputs[-(i + 1)])

        # Pass through segmentation head
        predicted_masks = self.segmentation_head(x) # 1 mask (channel) per class
            
        return predicted_masks