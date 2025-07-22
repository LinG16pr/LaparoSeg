import torch
import torch.nn as nn

from components.frame_encoder          import FrameEncoder
from components.memory_attention       import MemoryAttention
from components.mask_decoder           import MaskDecoder
from components.hierarchical_dense_crf import HierarchicalDenseCRF
from components.memory_encoder         import MemoryEncoder
from components.memory_bank            import MemoryBank



# ===============================[ DMBMSNet ]===============================

class DMBMSNet(nn.Module):

    def __init__(
        self,
        input_dim:          int,
        num_classes:        int,
        num_encoder_blocks: int  = 5,
        base_dim:           int  = 64,
        num_heads:          int  = 8,
        k:                  int  = 5,
        memory_frames:      int  = 4,
        num_memory_blocks:  int  = 4,
        deformable:         bool = True,
        use_crf:            bool = False
    ):
        super(DMBMSNet, self).__init__()

        self.base_dim = base_dim
        self.num_encoder_blocks = num_encoder_blocks
        self.use_crf = use_crf

        # Frame Encoder
        self.image_encoder = FrameEncoder(
            input_dim  = input_dim,
            num_blocks = num_encoder_blocks,
            base_dim   = base_dim,
            num_heads  = num_heads,
            k          = k,
            deformable = deformable
        )

        # Projection Layer for memory
        self.memory_projection = nn.Linear(
            in_features  = base_dim * (2 ** (num_encoder_blocks - 1)),
            out_features = base_dim * (2 ** (num_encoder_blocks - 1))
        )

        # Memory Attention
        self.memory_attention = MemoryAttention(
            dim           = base_dim * (2 ** (num_encoder_blocks - 1)),
            num_heads     = num_heads,
            memory_frames = memory_frames,
            num_blocks    = num_memory_blocks
        )

        # Mask Decoder
        self.mask_decoder = MaskDecoder(
            final_dim   = base_dim * (2 ** (num_encoder_blocks - 1)),
            num_classes = num_classes,
            num_blocks  = num_encoder_blocks - 1,
            base_dim    = base_dim,
            #num_heads   = num_heads,
            #k           = k,
            deformable  = deformable
        )

        # Hierarchical CRF
        if use_crf:
            self.hierarchical_dense_crf = HierarchicalDenseCRF(
                num_classes        = num_classes, 
                num_iterations     = 3,
                num_encoder_blocks = num_encoder_blocks, 
                base_dim           = base_dim
            )


        # Memory Encoder
        self.memory_encoder = MemoryEncoder(
            input_dim  = num_classes,
            target_dim = base_dim * (2 ** (num_encoder_blocks - 1)),
            num_layers = num_encoder_blocks - 1
        )

        # Memory Bank
        self.memory_bank = MemoryBank(
            max_frames = memory_frames
        )

    def forward(
        self,
        x:        torch.Tensor,
        training: bool = False,
        memory_frames  = None,
        memory_masks   = None
    ):

        # STEP 0: Verify inputs
        #

        # STEP 1: Pass through ImageEncoder

        encoder_output, skip_connections = self.image_encoder(x)

        # STEP 2: Flatten ImageEncoder output for MemoryAttention

        batch_size, C, H, W = encoder_output.shape
        encoder_output_flat = encoder_output.view(batch_size, H*W, C)

        # STEP 3: Retrieve memory
        
        if training:
            for i in range(len(memory_frames)):
                encoded_past_frame, _ = self.image_encoder(memory_frames[i])
                encoder_past_memory = self.memory_encoder(memory_masks[i], encoded_past_frame)
                self.memory_bank.add(encoder_past_memory)

        memory = self.memory_bank.get_memory(batch_size=x.shape[0], channels=self.base_dim * (2 ** (self.num_encoder_blocks - 1)), height=H, width=W)

        # STEP 4: Flatten memory spatial dimensions for MemoryAttention

        batch_size, num_frames, channels, height, width = memory.shape
        memory_flat = memory.view(batch_size, num_frames, channels, height*width).permute(0, 1, 3, 2)
        memory_flat = memory_flat.reshape(batch_size, num_frames*(height*width), channels)

        # STEP 5: Project memory to match ImageEncoder output dimensions

        memory_flat = self.memory_projection(memory_flat) # Shape: (batch_size, memory_frames * H*W, 256)

        # STEP 6: Pass through Memory Attention

        memory_attention_output = self.memory_attention(encoder_output_flat, memory_flat) # Shape: (batch_size, H*W, 256)

        # STEP 7: Reshape back to spatial dimensions for MaskDecoder

        memory_attention_output = memory_attention_output.view(batch_size, H, W, C).permute(0, 3, 1, 2)

        # STEP 8: Pass through MaskDecoder

        predicted_masks = self.mask_decoder(memory_attention_output, skip_connections) # Shape: (batch_size, num_classes, height, width)

        # Initialize pre_crf_masks for return when CRF is not used
        pre_crf_masks = predicted_masks

        # STEP 9: Pass through HierarchicalCRF

        if self.use_crf:
            post_crf_masks = self.hierarchical_dense_crf(predicted_masks, x, skip_connections, True)
        else:
            post_crf_masks = predicted_masks

        # STEP 10: Pass through MemoryEncoder + Update MemoryBank

        if not training:
            new_memory = self.memory_encoder(post_crf_masks, encoder_output)
            self.memory_bank.add(new_memory)

        # if self.use_crf:
        #     return pre_crf_masks, post_crf_masks
        return post_crf_masks