import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================[ HierarchicalDenseCRF ]===============================

class HierarchicalDenseCRF(nn.Module):

    def __init__(
        self, 
        num_classes:        int = 8, 
        num_iterations:     int = 2, 
        spatial_ksize:      int = 3, 
        bilateral_ksize:    int = 3, 
        num_encoder_blocks: int = 5, 
        base_dim:           int = 64, 
        skip_channels           = None
    ):
        """
        Hierarchical Dense CRF qui intègre les skip features de l'encodeur d'images.
        """
        super(HierarchicalDenseCRF, self).__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # Filtrage spatial : convolution depthwise par classe.
        self.spatial_conv = nn.Conv2d(
            num_classes, num_classes,
            kernel_size=spatial_ksize,
            padding=spatial_ksize // 2,
            groups=num_classes,
            bias=False
        )
        # Filtrage bilatéral : utilise la concaténation de Q et du guidance.
        self.bilateral_conv = nn.Conv2d(
            num_classes + 6, num_classes,
            kernel_size=bilateral_ksize,
            padding=bilateral_ksize // 2,
            bias=False
        )
        # Transformation de compatibilité : initialisée à -I.
        self.compat_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        nn.init.constant_(self.compat_conv.weight, 0)
        with torch.no_grad():
            self.compat_conv.weight += -torch.eye(num_classes).view(num_classes, num_classes, 1, 1)

        # Calcul dynamique des skip_channels si non fourni.
        if skip_channels is None:
            if num_encoder_blocks is None or base_dim is None:
                raise ValueError("Either provide skip_channels or both num_encoder_blocks and base_dim.")
            skip_channels = [base_dim * (2 ** i) for i in range(num_encoder_blocks - 1)]
        self.skip_channels = skip_channels

        # Projection des skip features vers 3 canaux.
        self.skip_proj = nn.ModuleList([nn.Conv2d(ch, 3, kernel_size=1) for ch in self.skip_channels])

    def fuse_skip_features(self, skip_features, target_size):
        """
        Projette et fusionne les skip features pour obtenir un guidance de 3 canaux.
        """
        if len(skip_features) != len(self.skip_proj):
            raise ValueError(f"Expected {len(self.skip_proj)} skip features, but got {len(skip_features)}.")
        
        proj_feats = []
        for feat, proj in zip(skip_features, self.skip_proj):
            feat_proj = proj(feat)
            feat_proj = F.interpolate(feat_proj, size=target_size, mode='bilinear', align_corners=True)
            proj_feats.append(feat_proj)
        fused = torch.stack(proj_feats, dim=0).mean(dim=0)
        return fused

    def mean_field_update(self, unary, guidance, final_logits=False):
        """
        Effectue les itérations du mean-field update.
        
        Args:
            unary (Tensor): Les logits non raffinés.
            guidance (Tensor): Guidance issue de l'image et des skip features.
            final_logits (bool): Si True, ne fait pas de softmax à la dernière itération.
        """
        Q = F.softmax(unary, dim=1)
        for i in range(self.num_iterations):
            spatial_msg = self.spatial_conv(Q)
            bilateral_input = torch.cat([Q, guidance], dim=1)
            bilateral_msg = self.bilateral_conv(bilateral_input)
            message = self.compat_conv(spatial_msg + bilateral_msg)
            updated = unary - message
            # Si c'est la dernière itération et que l'on souhaite obtenir des logits, on ne fait pas de softmax.
            if i == self.num_iterations - 1 and final_logits:
                Q = updated
            else:
                Q = F.softmax(updated, dim=1)
        return Q

    def forward(self, logits, image, skip_features=None, return_logits=False):
        """
        Args:
            logits (Tensor): Logits non raffinés provenant du Mask Decoder (N, num_classes, H, W).
            image (Tensor): Image d'entrée (N, 3, H, W).
            skip_features (list, optionnel): Liste des skip features de l'encodeur.
            return_logits (bool): Si True, renvoie des logits (scores bruts) au lieu des probabilités.
        """
        target_size = logits.shape[2:]
        if skip_features is not None:
            fused_skip = self.fuse_skip_features(skip_features, target_size)
        else:
            fused_skip = torch.zeros((image.shape[0], 3, *target_size), device=image.device)
        
        guidance = torch.cat([image, fused_skip], dim=1)  # (N, 6, H, W)
        
        # Mean-field update à pleine résolution
        Q_full = self.mean_field_update(logits, guidance, final_logits=return_logits)
        
        # Mise à l'échelle pour une meilleure cohérence globale
        logits_down = F.interpolate(logits, scale_factor=0.5, mode='bilinear', align_corners=True)
        guidance_down = F.interpolate(guidance, scale_factor=0.5, mode='bilinear', align_corners=True)
        Q_down = self.mean_field_update(logits_down, guidance_down, final_logits=return_logits)
        Q_down_up = F.interpolate(Q_down, size=target_size, mode='bilinear', align_corners=True)
        
        # Combinaison des deux échelles
        Q_combined = (Q_full + Q_down_up) / 2
        return Q_combined