import torch
import torch.nn as nn



# ===============================[ MemoryEncoder ]===============================

class MemoryEncoder(nn.Module):

    def __init__(
        self,
        input_dim:  int = 12,
        target_dim: int = 1024,
        num_layers: int = 4
    ):
        """
        The MemoryEncoder adjust the dimensions of the MaskDecoder output to fit the output of the ImageEncoder.

        Args:
            input_dim (int): Number of channels of the input (= number of channels of the MaskDecoder output, = number of classes).
            target_dim (int): Number of channels from the ImageEncoder output.
            num_layers (int): Number of convolutional layers for the MemoryEncoder.
        """
        super(MemoryEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        
        for _ in range(num_layers):
            # Convolutional layer to augment the number of channels and divide by a factor of 2 the height and width.
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_dim, current_dim * 2, kernel_size = 3, padding = 1, bias = False),
                nn.BatchNorm2d(current_dim * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 2, stride = 2)
            ))
            current_dim *= 2
        
        # Last layer to adjust final dimensions
        self.final_layer = nn.Conv2d(current_dim, target_dim, kernel_size = 1, bias = False)

    def forward(
        self,
        x:             torch.Tensor,
        encoded_frame: torch.Tensor
    ):
        """
        Forward propagation for the MemoryEncoder.

        Args:
            x (torch.Tensor): Input tensor with dimensions [batch_size, 1, height, width].

        Returns:
            torch.Tensor: Output tensor with dimensions [batch_size, target_dim, H', W'].
        """

        # Ensure that the input tensor contains floats
        x = x.float()

        # Re-add the channels dimension
        #x = x.unsqueeze(1) # Not needed if the masks are not fused...

        for layer in self.layers:
            x = layer(x)
        
        x = self.final_layer(x)

        # Fuse encoded frame with its relative encoded masks
        x = x + encoded_frame

        return x