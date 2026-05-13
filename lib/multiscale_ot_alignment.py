"""
Multi-Scale OT Feature Alignment for Enhanced_RRSIS_UOT.

Extends the single-scale OT alignment from RRSIS_SAM3 to operate across
multiple FPN (Feature Pyramid Network) levels with scale-aware projections.

Key idea:
    - Different FPN levels capture objects at different scales
    - Small objects benefit from fine-grained (high-res) alignment
    - Large objects benefit from coarse (low-res) alignment
    - Each scale gets its own OT alignment with scale-specific projections
    - A learnable scale weighting combines the aligned features

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
    Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareOTAligner(nn.Module):
    """
    OT-based feature aligner for a single FPN scale with scale-specific
    projection heads.

    Args:
        d_model: Hidden dimension.
        reg: Sinkhorn entropy regularization.
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Scale factor for the OT-aligned text residual.
        scale_id: Index of this scale (for logging/identification).
    """

    def __init__(self, d_model, reg=0.1, num_iter=10, residual_weight=0.5, scale_id=0):
        super().__init__()
        self.d_model = d_model
        self.reg = reg
        self.num_iter = num_iter
        self.residual_weight = residual_weight
        self.scale_id = scale_id

        # Scale-specific projection heads
        self.text_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Scale-specific gating: learns how much OT-aligned text to add
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Initialize output projection near zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @torch.no_grad()
    def sinkhorn(self, cost_matrix):
        """
        Balanced Sinkhorn algorithm for optimal transport.

        Args:
            cost_matrix: (B, N_img, N_txt) pairwise cost.

        Returns:
            Transport plan P of shape (B, N_img, N_txt).
        """
        B, N, M = cost_matrix.shape

        # Uniform marginals
        mu = torch.full((B, N), 1.0 / N, device=cost_matrix.device, dtype=cost_matrix.dtype)
        nu = torch.full((B, M), 1.0 / M, device=cost_matrix.device, dtype=cost_matrix.dtype)

        # Gibbs kernel
        K = torch.exp(-cost_matrix / self.reg)
        K = K.clamp(min=1e-8)  # Numerical stability

        u = torch.ones_like(mu)
        for _ in range(self.num_iter):
            v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            u = mu / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)

        # Transport plan
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        return P

    def forward(self, img_feat, text_feat, text_mask=None):
        """
        Scale-specific OT alignment.

        Args:
            img_feat: (B, C, H, W) image features.
            text_feat: (seq, B, C) or (B, seq, C) text features.
            text_mask: (B, seq) boolean mask, True = padding.

        Returns:
            aligned_img: (B, C, H, W) image features enhanced with
                         OT-aligned text at this scale.
        """
        B, C, H, W = img_feat.shape

        # Reshape image to (B, H*W, C)
        img_flat = img_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Text: handle both (seq, B, C) and (B, seq, C)
        if text_feat.dim() == 3 and text_feat.shape[1] == B and text_feat.shape[0] != B:
            text_flat = text_feat.permute(1, 0, 2)  # (B, seq, C)
        elif text_feat.dim() == 3:
            text_flat = text_feat  # Already (B, seq, C)
        else:
            return img_feat  # Can't align without proper text shape

        # Mask out padding text tokens
        if text_mask is not None:
            valid_mask = ~text_mask
            text_flat = text_flat * valid_mask.unsqueeze(-1).float()

        # Project to alignment space
        img_proj = self.img_proj(img_flat)
        txt_proj = self.text_proj(text_flat)

        # Cost matrix via negative cosine similarity
        img_norm = F.normalize(img_proj, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        cost = 1.0 - torch.bmm(img_norm, txt_norm.transpose(1, 2))  # (B, HW, seq)

        # Compute OT plan
        P = self.sinkhorn(cost)  # (B, HW, seq)

        # Transport text to image positions
        aligned_text = torch.bmm(P * P.shape[1], text_flat)  # (B, HW, C)
        aligned_text = self.output_proj(aligned_text)

        # Gated residual: learn how much aligned text to mix in
        gate_input = torch.cat([img_flat, aligned_text], dim=-1)  # (B, HW, 2C)
        gate_weight = self.gate(gate_input)  # (B, HW, C) — per-position gate

        img_enhanced = img_flat + gate_weight * aligned_text
        img_enhanced = self.norm(img_enhanced)

        # Reshape back
        img_enhanced = img_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return img_enhanced


class MultiScaleOTAligner(nn.Module):
    """
    Multi-Scale OT Feature Alignment across all FPN levels.

    Creates a scale-specific OT aligner for each FPN level and applies
    them in parallel, then combines via learnable scale weighting.

    Args:
        d_model: Hidden dimension.
        num_scales: Number of FPN levels to align (default 3).
        reg: Sinkhorn entropy regularization.
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Residual weight for each scale.
    """

    def __init__(self, d_model, num_scales=3, reg=0.1, num_iter=10, residual_weight=0.5):
        super().__init__()
        self.num_scales = num_scales

        # Create a scale-specific aligner for each FPN level
        self.scale_aligners = nn.ModuleList([
            ScaleAwareOTAligner(
                d_model=d_model,
                reg=reg,
                num_iter=num_iter,
                residual_weight=residual_weight,
                scale_id=i,
            )
            for i in range(num_scales)
        ])

        # Learnable scale weighting: how much each scale contributes
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        print(f"[MultiScaleOT] Initialized {num_scales}-scale OT alignment (d_model={d_model})")

    def forward(self, fpn_features, text_feat, text_mask=None):
        """
        Apply scale-aware OT alignment to each FPN level.

        Args:
            fpn_features: List of FPN feature maps, each (B, C, H_i, W_i).
            text_feat: (seq, B, C) text features.
            text_mask: (B, seq) boolean mask, True = padding.

        Returns:
            aligned_fpn: List of aligned FPN feature maps.
        """
        aligned_fpn = []
        weights = F.softmax(self.scale_weights, dim=0)

        for i, feat in enumerate(fpn_features):
            # Handle both plain tensors and NestedTensor
            if hasattr(feat, 'tensors'):
                feat_tensor = feat.tensors
            else:
                feat_tensor = feat

            if feat_tensor.dim() != 4:
                aligned_fpn.append(feat)
                continue

            # Use the aligner for this scale (or last one if we have more FPN levels)
            aligner_idx = min(i, len(self.scale_aligners) - 1)
            aligner = self.scale_aligners[aligner_idx]

            # Apply scale-specific OT alignment
            aligned = aligner(feat_tensor, text_feat, text_mask)

            # Weighted blend: original + weight * (aligned - original)
            w = weights[aligner_idx]
            blended = (1 - w) * feat_tensor + w * aligned

            if hasattr(feat, 'tensors'):
                feat.tensors = blended
                aligned_fpn.append(feat)
            else:
                aligned_fpn.append(blended)

        return aligned_fpn
