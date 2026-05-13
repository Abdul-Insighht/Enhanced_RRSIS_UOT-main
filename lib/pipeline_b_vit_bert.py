import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticPrompterFeatureReuse(nn.Module):
    """
    Pipeline B: Semantic Referring Prompter (Optimized Feature-Reuse version).
    
    This module acts as a secondary pipeline that:
    1. Re-uses the extracted Multi-Scale visual features and Text features from the SAM3 backbone.
    2. Projects them to a common subspace if needed.
    3. Fuses them using Cross-Attention to locate the object (slack localization).
    4. Generates Sparse (point-like) and Dense (pseudo-mask) prompts for SAM3 Decoder.
    """
    def __init__(
        self,
        image_size=504,
        decoder_hidden_dim=256,
        visual_feat_dim=256,
        text_feat_dim=256,
        dense_prompt_side=16, # Controls resolution of dense prompt
        num_sparse_prompts=4,
    ):
        super().__init__()
        self.image_size = image_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dense_prompt_side = dense_prompt_side
        self.num_sparse_prompts = num_sparse_prompts

        # 1. Projection Heads to Common Subspace
        self.common_dim = decoder_hidden_dim
        self.vis_proj = nn.Linear(visual_feat_dim, self.common_dim) if visual_feat_dim != self.common_dim else nn.Identity()
        self.txt_proj = nn.Linear(text_feat_dim, self.common_dim) if text_feat_dim != self.common_dim else nn.Identity()

        # 2. Cross-Attention Fusion (Text queries Image)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4, batch_first=True)

        # 3. Prompt Generators
        # Sparse prompts (like learned tokens that act as points/boxes)
        self.sparse_prompt_generator = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim),
            nn.ReLU(),
            nn.Linear(self.common_dim, num_sparse_prompts * self.common_dim)
        )
        
        # Dense prompt (pseudo-mask embedding)
        # B x Common_Dim -> Reshape to spatial -> Conv Transpose -> B x Decoder_Dim x 16 x 16
        self.dense_prompt_generator = nn.Sequential(
            nn.ConvTranspose2d(self.common_dim, self.common_dim, kernel_size=2, stride=2), # 2x upsample
            nn.BatchNorm2d(self.common_dim),
            nn.ReLU(),
            nn.Conv2d(self.common_dim, decoder_hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, vis_feats, txt_feats):
        """
        Args:
            vis_feats: [B, C, H, W] or [B, N, C] tensor from SAM3 backbone
            txt_feats: [seq_len, B, C] tensor from SAM3 text encoder
        Returns:
            dict containing visual_prompt_embed (dense) and sparse_prompts
        """
        B = txt_feats.shape[1] if txt_feats.dim() == 3 else vis_feats.shape[0]
        device = txt_feats.device

        # --- 1. Process Image Features ---
        if vis_feats.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B_v, C_v, H, W = vis_feats.shape
            vis_feats_seq = vis_feats.flatten(2).permute(0, 2, 1) # [B, N, C]
        elif hasattr(vis_feats, 'tensors'):
            # Handle NestedTensor if applicable
            v_tensor = vis_feats.tensors
            B_v, C_v, H, W = v_tensor.shape
            vis_feats_seq = v_tensor.flatten(2).permute(0, 2, 1)
        else:
            vis_feats_seq = vis_feats # Assume [B, N, C]
            
        vis_feats_proj = self.vis_proj(vis_feats_seq) # [B, N, C_common]

        # --- 2. Process Text Features ---
        # If [seq, B, C], change to [B, seq, C]
        if txt_feats.dim() == 3 and txt_feats.shape[1] == B:
            txt_feats_seq = txt_feats.permute(1, 0, 2)
        else:
            txt_feats_seq = txt_feats # Assume [B, seq, C]
            
        txt_feats_proj = self.txt_proj(txt_feats_seq) # [B, seq, C_common]

        # --- 3. Cross-Attention Fusion ---
        # Text attends to Image to find relevant regions
        fused_feats, _ = self.cross_attn(
            query=txt_feats_proj, 
            key=vis_feats_proj, 
            value=vis_feats_proj, 
            key_padding_mask=None
        ) # [B, seq_len, C_common]
        
        # Pool the fused text features into a single vector per batch
        pooled_fused = fused_feats.mean(dim=1) # [B, C_common]

        # --- 4. Generate Prompts for SAM3 ---
        # Sparse Prompts (Points)
        sparse_prompts = self.sparse_prompt_generator(pooled_fused) # [B, num_sparse * C]
        sparse_prompts = sparse_prompts.view(B, self.num_sparse_prompts, self.common_dim)

        # Dense Prompts (Pseudo-Mask)
        # Start with a 1x1 spatial feature and upsample it
        dense_base = pooled_fused.view(B, self.common_dim, 1, 1)
        # Let's upsample it directly to the target spatial size to save layers
        dense_base = F.interpolate(dense_base, size=(self.dense_prompt_side // 2, self.dense_prompt_side // 2), mode='nearest')
        dense_prompts = self.dense_prompt_generator(dense_base) # [B, decoder_dim, side, side]
        
        return {
            "sparse_prompts": sparse_prompts,
            "visual_prompt_embed": dense_prompts,
        }

    def build_visual_prompts_for_sam3(self, vis_feats, txt_feats):
        """Helper to generate and format prompts specifically for SAM3 injection."""
        outputs = self.forward(vis_feats, txt_feats)
        
        dense_prompt = outputs["visual_prompt_embed"] # [B, C, H, W]
        
        # Format for SAM3's prompt encoder: SAM3 expects visual_prompt_embed as sequence (L, B, C)
        B, C, H, W = dense_prompt.shape
        dense_seq = dense_prompt.flatten(2).permute(2, 0, 1) # [H*W, B, C]
        
        # Mask is false everywhere (we use all tokens)
        prompt_mask = torch.zeros((B, H*W), dtype=torch.bool, device=dense_prompt.device)

        return {
            "visual_prompt_embed": dense_seq,
            "visual_prompt_mask": prompt_mask,
            "sparse_prompts": outputs["sparse_prompts"] # Can be used as extra points if needed
        }
