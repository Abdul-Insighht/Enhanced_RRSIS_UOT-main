"""
OT-Based Feature Alignment: Uses Sinkhorn algorithm to compute optimal
transport between text tokens and image spatial tokens for better
cross-modal fusion before the encoder.

Instead of SAM3's default approach of mean-pooling text and adding uniformly
to all image features, this module computes a soft OT plan so each image
spatial position receives a unique weighted combination of text tokens.

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTFeatureAligner(nn.Module):
    """
    Computes Sinkhorn-based optimal transport between text and image tokens
    to produce spatially-varying text features aligned to image features.

    Each image spatial position receives a text representation weighted by
    its OT coupling, rather than a uniform mean-pool.

    Args:
        d_model: Hidden dimension of both text and image features.
        reg: Sinkhorn entropy regularization (lower = sharper matching).
        num_iter: Number of Sinkhorn iterations.
        residual_weight: Scale factor for the OT-aligned text residual.
    """

    def __init__(self, d_model, reg=0.1, num_iter=10, residual_weight=0.5):
        super().__init__()
        self.d_model = d_model
        self.reg = reg
        self.num_iter = num_iter
        self.residual_weight = residual_weight

        # Project text and image into shared alignment space
        self.text_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Initialize output projection near zero so residual starts small
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @torch.no_grad()
    def sinkhorn(self, cost_matrix):
        """
        Balanced Sinkhorn algorithm for OT.

        Args:
            cost_matrix: (B, N_img, N_txt) pairwise cost.

        Returns:
            Transport plan P of shape (B, N_img, N_txt).
        """
        B, N, M = cost_matrix.shape

        # Uniform marginals
        mu = torch.full((B, N), 1.0 / N, device=cost_matrix.device)
        nu = torch.full((B, M), 1.0 / M, device=cost_matrix.device)

        # Gibbs kernel
        K = torch.exp(-cost_matrix / self.reg)

        u = torch.ones_like(mu)
        for _ in range(self.num_iter):
            v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            u = mu / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)

        # Transport plan: P = diag(u) @ K @ diag(v)
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        return P

    def forward(self, img_feat, text_feat, text_mask=None):
        """
        Args:
            img_feat: (B, C, H, W) image features from backbone FPN.
            text_feat: (seq_len, B, C) text features (seq-first format).
            text_mask: (B, seq_len) boolean mask, True = padding token.

        Returns:
            aligned_img: (B, C, H, W) image features enhanced with
                         OT-aligned text information.
        """
        B, C, H, W = img_feat.shape

        # Reshape image to (B, H*W, C)
        img_flat = img_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Text: (seq, B, C) → (B, seq, C)
        text_flat = text_feat.permute(1, 0, 2)  # (B, seq, C)

        # Mask out padding text tokens
        if text_mask is not None:
            valid_mask = ~text_mask  # True = valid
            text_flat = text_flat * valid_mask.unsqueeze(-1).float()

        # Project to alignment space
        img_proj = self.img_proj(img_flat)     # (B, HW, C)
        txt_proj = self.text_proj(text_flat)   # (B, seq, C)

        # Compute cost matrix as negative cosine similarity
        img_norm = F.normalize(img_proj, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        cost = 1.0 - torch.bmm(img_norm, txt_norm.transpose(1, 2))  # (B, HW, seq)

        # Compute OT plan (stop gradients through Sinkhorn iterations)
        P = self.sinkhorn(cost)  # (B, HW, seq)

        # Transport text features to image positions
        # Scale by N so each position gets ~1 unit of text mass
        aligned_text = torch.bmm(P * P.shape[1], text_flat)  # (B, HW, C)
        aligned_text = self.output_proj(aligned_text)

        # Residual addition
        img_enhanced = img_flat + self.residual_weight * aligned_text
        img_enhanced = self.norm(img_enhanced)

        # Reshape back to spatial format
        img_enhanced = img_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return img_enhanced
