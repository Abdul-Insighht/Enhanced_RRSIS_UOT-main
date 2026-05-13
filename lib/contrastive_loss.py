"""
Contrastive Language-Image Loss (InfoNCE) for Enhanced_RRSIS_UOT.

Adds an auxiliary contrastive learning objective that enforces alignment
between the visual features of the masked (segmented) region and the
corresponding text features. This helps the model learn tighter
cross-modal representations.

Key idea:
    - Pool visual features from the predicted mask region
    - Pool text features from the text encoder output
    - Apply InfoNCE contrastive loss within the batch:
        Positive pair = (masked visual, matching text)
        Negative pairs = (masked visual, non-matching texts in batch)

Reference:
    Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision" (CLIP), ICML 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for vision-language alignment.

    Computes bidirectional contrastive loss between masked region visual
    features and text features within a mini-batch.

    Args:
        temperature: Scaling temperature for logits (lower = sharper).
        visual_dim: Dimension of visual features.
        text_dim: Dimension of text features.
        proj_dim: Shared projection dimension for alignment.
    """

    def __init__(self, temperature=0.07, visual_dim=256, text_dim=256, proj_dim=128):
        super().__init__()
        self.temperature = temperature

        # Project visual and text features into shared space
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Learnable temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

    def mask_pool(self, features, masks):
        """
        Pool visual features from masked region using masked average pooling.

        Args:
            features: (B, C, H, W) visual feature maps.
            masks: (B, 1, H, W) binary masks (predicted or GT).

        Returns:
            (B, C) pooled visual features from the masked region.
        """
        # Resize mask to match feature spatial dimensions if needed
        if features.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(
                masks.float(), features.shape[-2:], mode='nearest'
            )

        # Ensure mask is binary
        masks = (masks > 0.5).float()

        # Masked average pooling
        # (B, C, H, W) * (B, 1, H, W) → (B, C, H, W) → sum over spatial → (B, C)
        masked_features = features * masks
        pooled = masked_features.sum(dim=(2, 3))

        # Denominator: sum of mask values per sample
        mask_sum = masks.sum(dim=(2, 3)).clamp(min=1.0)  # (B, 1)
        pooled = pooled / mask_sum

        return pooled

    def forward(self, visual_features, text_features, pred_masks, gt_masks=None):
        """
        Compute bidirectional InfoNCE contrastive loss.

        Args:
            visual_features: (B, C, H, W) image feature maps from backbone/encoder.
            text_features: (B, C_text) or (seq, B, C_text) text features.
            pred_masks: (B, 1, H, W) predicted mask logits.
            gt_masks: (B, 1, H, W) ground truth masks (used for pooling if available).

        Returns:
            contrastive_loss: scalar tensor.
        """
        B = visual_features.shape[0]

        # Use GT masks for pooling during training (more stable)
        # Fall back to predicted masks during inference
        pool_masks = gt_masks if gt_masks is not None else torch.sigmoid(pred_masks)

        # Pool visual features from masked region
        visual_pooled = self.mask_pool(visual_features, pool_masks)  # (B, C)

        # Handle text features shape
        if text_features.dim() == 3:
            # (seq, B, C) → (B, C) via mean pooling over sequence
            text_pooled = text_features.mean(dim=0)
        else:
            text_pooled = text_features  # Already (B, C)

        # Project to shared space
        visual_emb = F.normalize(self.visual_proj(visual_pooled), dim=-1)  # (B, D)
        text_emb = F.normalize(self.text_proj(text_pooled), dim=-1)  # (B, D)

        # If batch size is 1, contrastive loss is meaningless
        if B <= 1:
            return torch.tensor(0.0, device=visual_features.device, requires_grad=True)

        # Compute similarity matrix: (B, B)
        temp = self.log_temp.exp().clamp(min=0.01, max=100.0)
        logits = torch.mm(visual_emb, text_emb.T) / temp

        # Labels: diagonal = positive pairs
        labels = torch.arange(B, device=logits.device)

        # Bidirectional InfoNCE loss
        loss_v2t = F.cross_entropy(logits, labels)  # Visual → Text
        loss_t2v = F.cross_entropy(logits.T, labels)  # Text → Visual

        return (loss_v2t + loss_t2v) / 2.0
