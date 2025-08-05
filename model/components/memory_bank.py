import torch



# ===============================[ MemoryBank ]===============================

class MemoryBank:

    def __init__(
        self,
        max_frames: int
    ):
        """
        MemoryBank manages a set of frames with dimensions [batch_size, 1024, height/16, width/16].
        Args:
            max_frames (int): Maximum number of frames to remember.
            device (str): Device to store the frames (e.g., 'cuda' ou 'cpu').
        """
        self.max_frames = max_frames
        self.bank = []

    def add(
        self,
        frame: torch.Tensor
    ):
        """
        Add a new frame to the MemoryBank.
        Args:
            frame (torch.Tensor): Frame with dimensions [batch_size, 1024, height/16, width/16].
        """
        if len(self.bank) >= self.max_frames and len(self.bank) != 0:
            self.bank.pop(0)  # Delete the oldest frame
        self.bank.append(frame)  # Add the new frame

    def get_memory(
        self,
        batch_size: int,
        channels:   int,
        height:     int,
        width:      int
    ):
        """
        Get MemoryBank content as a tensor.
        If MemoryBank is empty, returns a tensor of zeros.
        Args:
            batch_size (int): Number of frames per batch.
            channels (int): Number of channels (1024 by default).
            height (int): Frames height in pixels (H/16).
            width (int): Frames width in pixels (W/16).
        Returns:
            torch.Tensor: Tensor with dimensions [batch_size, max_frames, channels, height, width].
        """
        
        # If empty memory
        if len(self.bank) == 0:
            # Return a tensor of zeros
            return torch.zeros((batch_size, self.max_frames, channels, height, width), device="cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"), requires_grad=True)
        
        memory = torch.stack(self.bank, dim=1)  # Shape: (batch_size, len(self.bank), channels, height, width)
        
        # Uncomplete memory
        if len(self.bank) < self.max_frames:
            pad_frames = self.max_frames - len(self.bank)
            oldest_frame = memory[:, 0:1, :, :, :] # Shape: (batch_size, pad_frames, C, H, W)
            padding = oldest_frame.repeat(1, pad_frames, 1, 1, 1) # Shape: (batch_size, pad_frames, C, H, W)
            memory = torch.cat([padding, memory], dim=1) # Final shape: (batch_size, max_frames, C, H, W)

        return memory
    
    def clear(self):
        self.bank = []