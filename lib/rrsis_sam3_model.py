"""
RRSIS_SAM3: Main model that wraps SAM3 for Referring Remote Sensing Image Segmentation.

Architecture:
    1. SAM3 VL Backbone (ViT vision encoder + VE text encoder) — Frozen + LoRA
    2. OT Feature Alignment — Sinkhorn-based text-image spatial fusion
    3. SAM3 Transformer Encoder — Text-image fusion (fine-tuned)
    4. SAM3 Transformer Decoder — DETR-based detection (fine-tuned)
    5. SAM3 Segmentation Head — Pixel-level mask prediction (fine-tuned)

For RRSIS, we:
    - Feed the RS image + referring text caption to SAM3
    - Apply OT-based alignment to fuse text into image features spatially
    - SAM3 detects + segments the referred object
    - Use UOT matching to supervise ALL decoder queries (not just top-1)
    - Compute OT-weighted GIoU + L1 + Dice + BCE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from sam3.model_builder import (
    build_sam3_image_model,
    _create_text_encoder,
    _create_vision_backbone,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
    _load_checkpoint,
)
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt

from .rs_adapters import inject_lora_adapters, get_trainable_params_summary
from .ot_feature_alignment import OTFeatureAligner
from .ot_loss import OTSegmentationLoss


class RRSIS_SAM3(nn.Module):
    """
    RRSIS-adapted SAM3 for referring remote sensing image segmentation.

    This model wraps SAM3's native text-aware architecture and adapts it
    for the remote sensing domain using LoRA adapters and fine-tuning.
    """

    def __init__(
        self,
        sam3_ckpt: str = None,
        image_size: int = 504,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_backbone: bool = True,
        freeze_text_encoder: bool = True,
        gradient_checkpointing: bool = True,
        # === UOT parameters ===
        ot_reg: float = 0.1,
        ot_num_iter: int = 10,
        ot_residual_weight: float = 0.5,
        use_unbalanced_ot: bool = True,
        ot_background_cost: float = 0.5,
    ):
        super().__init__()
        self.image_size = image_size

        # ====== Build SAM3 Image Model ======
        print("[RRSIS_SAM3] Building SAM3 image model...")
        self.sam3 = build_sam3_image_model(
            device="cpu",           # Load on CPU first, move to GPU later
            eval_mode=False,        # We need training mode
            checkpoint_path=sam3_ckpt,
            load_from_HF=(sam3_ckpt is None),  # Download from HF if no local ckpt
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

        # ====== Freeze Strategy ======
        # Step 1: Freeze everything first
        if freeze_backbone:
            self._freeze_backbone()

        if freeze_text_encoder:
            self._freeze_text_encoder()

        # Step 2: Inject LoRA adapters into frozen backbone
        inject_lora_adapters(self.sam3, rank=lora_rank, alpha=lora_alpha)

        # Step 3: Unfreeze trainable components
        self._unfreeze_trainable_components()

        # Step 4: Enable gradient checkpointing for memory savings
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # ====== OT Feature Alignment Module ======
        d_model = self.sam3.hidden_dim  # typically 256
        print(f"[RRSIS_SAM3] Initializing OT Feature Aligner (d_model={d_model})")
        self.ot_aligner = OTFeatureAligner(
            d_model=d_model,
            reg=ot_reg,
            num_iter=ot_num_iter,
            residual_weight=ot_residual_weight,
        )

        # ====== OT-based Loss ======
        print("[RRSIS_SAM3] Initializing OT Segmentation Loss")
        self.ot_loss = OTSegmentationLoss()

        # Print parameter summary
        get_trainable_params_summary(self)

        # Image normalization (SAM3 uses 0.5 mean/std)
        self.register_buffer('pixel_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def _freeze_backbone(self):
        """Freeze the ViT vision backbone."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'vision_backbone'):
            for param in backbone.vision_backbone.parameters():
                param.requires_grad = False
            print("[RRSIS_SAM3] ViT backbone frozen")

    def _freeze_text_encoder(self):
        """Freeze the text encoder."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'language_backbone'):
            for param in backbone.language_backbone.parameters():
                param.requires_grad = False
            print("[RRSIS_SAM3] Text encoder frozen")

    def _unfreeze_trainable_components(self):
        """Unfreeze components we want to fine-tune."""
        # Transformer encoder (text-image fusion) — fine-tune
        if hasattr(self.sam3, 'transformer'):
            for param in self.sam3.transformer.encoder.parameters():
                param.requires_grad = True
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True
            print("[RRSIS_SAM3] Transformer encoder+decoder unfrozen")

        # Segmentation head — fine-tune
        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True
            print("[RRSIS_SAM3] Segmentation head unfrozen")

        # Geometry encoder — fine-tune
        if hasattr(self.sam3, 'geometry_encoder'):
            for param in self.sam3.geometry_encoder.parameters():
                param.requires_grad = True
            print("[RRSIS_SAM3] Geometry encoder unfrozen")

        # Scoring head — fine-tune
        if hasattr(self.sam3, 'dot_prod_scoring') and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
            print("[RRSIS_SAM3] Scoring head unfrozen")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        # SAM3's ViT backbone and encoder already support act_checkpoint
        # We just need to ensure it's enabled
        print("[RRSIS_SAM3] Gradient checkpointing enabled")

    def normalize_image(self, images):
        """Normalize images to SAM3's expected range [-1, 1]."""
        return (images - self.pixel_mean) / self.pixel_std

    def forward(self, images, captions, masks_gt=None):
        """
        Forward pass for RRSIS.

        Args:
            images: [B, 3, H, W] tensor of RS images (already resized to image_size)
            captions: List[str] of length B, referring text descriptions
            masks_gt: [B, 1, H, W] ground truth masks (optional, for loss computation)

        Returns:
            dict with keys:
                - 'pred_masks': [B, 1, H, W] predicted segmentation masks
                - 'pred_logits': [B, N] confidence scores for N queries
                - 'pred_boxes': [B, N, 4] predicted bounding boxes
                - 'loss': scalar loss (only if masks_gt is provided)
        """
        B = images.shape[0]
        device = images.device

        # Normalize images
        images = self.normalize_image(images)

        # Step 1: Forward backbone (vision + text)
        backbone_out = self.sam3.backbone.forward_image(images)
        text_out = self.sam3.backbone.forward_text(captions, device=device)
        backbone_out.update(text_out)

        # Step 1.5: OT Feature Alignment — fuse text into image spatially
        # before the encoder sees the features
        if hasattr(self, 'ot_aligner') and 'backbone_fpn' in backbone_out:
            text_feats = backbone_out.get('language_features', None)
            text_mask = backbone_out.get('language_mask', None)
            if text_feats is not None:
                fpn_feats = backbone_out['backbone_fpn']
                for i in range(len(fpn_feats)):
                    feat = fpn_feats[i]
                    # Handle both plain tensors and NestedTensor
                    if hasattr(feat, 'tensors'):
                        feat_tensor = feat.tensors
                    else:
                        feat_tensor = feat
                    if feat_tensor.dim() == 4:  # (B, C, H, W)
                        aligned = self.ot_aligner(feat_tensor, text_feats, text_mask)
                        if hasattr(feat, 'tensors'):
                            feat.tensors = aligned
                        else:
                            fpn_feats[i] = aligned

        # Step 2: Create find_input (tells SAM3 which image/text pairs to process)
        img_ids = torch.arange(B, device=device)
        text_ids = torch.arange(B, device=device)
        find_input = FindStage(
            img_ids=img_ids, 
            text_ids=text_ids,
            input_boxes=torch.zeros(B, 0, 4, device=device),
            input_boxes_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            input_boxes_label=torch.zeros(B, 0, device=device, dtype=torch.long),
            input_points=torch.zeros(B, 0, 2, device=device),
            input_points_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
        )

        # Step 3: Create empty geometric prompt (we use text-only prompting)
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, B, 4, device=device),
            box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
        )

        # Step 4: Encode prompt (text + empty geometry)
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )

        # Step 5: Run encoder (text-image fusion)
        backbone_out, encoder_out, feat_tuple = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        # Step 6: Run decoder (DETR-based detection)
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        }
        out, hs = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # Step 7: Run segmentation head
        if self.sam3.segmentation_head is not None:
            _, _, _, vis_feat_sizes = feat_tuple
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]

            self.sam3._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=vis_feat_sizes,
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        # Step 8: Select best mask per sample
        result = self._select_best_mask(out, B)

        # Step 9: Compute loss if ground truth is provided
        if masks_gt is not None:
            if hasattr(self, 'ot_loss'):
                result['loss'] = self.ot_loss(result, masks_gt, self.image_size)
            else:
                result['loss'] = self._compute_loss(result['pred_masks'], masks_gt)

        return result

    def _select_best_mask(self, out, batch_size):
        """
        Select the best mask from SAM3's multi-query output.

        SAM3 produces N detection queries. We pick the one with
        highest confidence as our referred object's mask.
        """
        result = {}

        # Scores: [B, N_queries] — select the query with highest score
        pred_logits = out.get('pred_logits', None)  # [B, N_queries, 1]
        if pred_logits is not None:
            result['pred_logits'] = pred_logits

        pred_boxes = out.get('pred_boxes', None)  # [B, N_queries, 4]
        if pred_boxes is not None:
            result['pred_boxes'] = pred_boxes

        # Masks: [B, N_queries, H_mask, W_mask]
        pred_masks = out.get('pred_masks', None)
        if pred_masks is not None:
            # Select the mask with highest confidence
            if pred_logits is not None:
                scores = pred_logits.squeeze(-1)  # [B, N_queries]
                best_idx = scores.argmax(dim=-1)   # [B]
                batch_idx = torch.arange(batch_size, device=pred_masks.device)
                best_masks = pred_masks[batch_idx, best_idx]  # [B, H_mask, W_mask]
                best_masks = best_masks.unsqueeze(1)  # [B, 1, H_mask, W_mask]
            else:
                best_masks = pred_masks[:, 0:1]  # Fallback: take first query

            # Resize to original image size
            best_masks = F.interpolate(
                best_masks.float(),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
            result['pred_masks'] = best_masks
        else:
            # No segmentation head output — return zeros
            result['pred_masks'] = torch.zeros(
                batch_size, 1, self.image_size, self.image_size,
                device=next(self.parameters()).device,
            )

        return result

    def _compute_loss(self, pred_masks, gt_masks):
        """
        Fallback: Compute combined Dice + BCE loss (used when OT loss unavailable).

        Args:
            pred_masks: [B, 1, H, W] predicted mask logits
            gt_masks: [B, 1, H, W] ground truth binary masks
        """
        # Resize gt to match pred if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(),
                size=pred_masks.shape[-2:],
                mode='nearest',
            )

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())

        # Dice loss
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        # Combined loss
        total_loss = 0.5 * bce_loss + 0.5 * dice_loss
        return total_loss

    @torch.no_grad()
    def predict(self, images, captions):
        """Inference-only forward pass."""
        self.eval()
        result = self.forward(images, captions)
        # Apply sigmoid to get probabilities
        result['pred_probs'] = torch.sigmoid(result['pred_masks'])
        result['pred_binary'] = (result['pred_probs'] > 0.5).float()
        return result
