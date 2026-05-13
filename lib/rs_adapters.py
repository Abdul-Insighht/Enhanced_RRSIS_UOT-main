"""
LoRA and Domain Adapters for adapting SAM3 to Remote Sensing domain.

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into frozen
transformer attention layers, allowing domain-specific fine-tuning with
minimal extra parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for linear projections.

    Decomposes weight update ΔW into two low-rank matrices: W = W0 + (α/r) * B @ A
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, and r << min(d_in, d_out).
    """

    def __init__(self, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros (so ΔW starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # ΔW @ x = B @ A @ x, scaled by α/r
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA adapter.
    The original linear is frozen; only LoRA params are trainable.
    """

    def __init__(self, original_linear, rank=16, alpha=32.0):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
        )
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


class LoRAMultiheadAttention(nn.Module):
    """
    Wraps nn.MultiheadAttention with LoRA on Q and V projections.
    """

    def __init__(self, original_mha, rank=16, alpha=32.0):
        super().__init__()
        self.original_mha = original_mha
        embed_dim = original_mha.embed_dim

        # LoRA on Q and V projections
        self.lora_q = LoRALayer(embed_dim, embed_dim, rank=rank, alpha=alpha)
        self.lora_v = LoRALayer(embed_dim, embed_dim, rank=rank, alpha=alpha)

        # Freeze original MHA weights
        for param in self.original_mha.parameters():
            param.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        # Add LoRA deltas to query and value
        query_delta = self.lora_q(query)
        value_delta = self.lora_v(value)

        return self.original_mha(
            query + query_delta,
            key,
            value + value_delta,
            **kwargs,
        )


def inject_lora_adapters(model, rank=16, alpha=32.0, target_modules=None):
    """
    Inject LoRA adapters into a SAM3 model's ViT backbone.

    This function finds attention layers in the ViT backbone and wraps their
    Q, K, V projections with LoRA layers, while freezing the original weights.

    Args:
        model: The SAM3 model (Sam3Image)
        rank: LoRA rank (default 16)
        alpha: LoRA alpha scaling factor
        target_modules: List of module name patterns to target (default: attention layers)

    Returns:
        Number of LoRA parameters added
    """
    lora_params = 0

    # Target the ViT backbone's attention layers
    backbone = model.backbone
    if hasattr(backbone, 'vision_backbone'):
        vit = backbone.vision_backbone
        # SAM3's ViT backbone is a Sam3DualViTDetNeck → trunk is the ViT
        if hasattr(vit, 'trunk'):
            trunk = vit.trunk
            # Iterate through ViT blocks and inject LoRA into attention
            if hasattr(trunk, 'blocks'):
                for i, block in enumerate(trunk.blocks):
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        # LoRA on qkv projection
                        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
                            original = attn.qkv
                            attn.qkv = LoRALinear(original, rank=rank, alpha=alpha)
                            lora_params += rank * original.in_features + original.out_features * rank
                        # LoRA on output projection
                        if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
                            original = attn.proj
                            attn.proj = LoRALinear(original, rank=rank, alpha=alpha)
                            lora_params += rank * original.in_features + original.out_features * rank

    # Also inject LoRA into the transformer encoder's cross-attention
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
        encoder = model.transformer.encoder
        if hasattr(encoder, 'layers'):
            for layer in encoder.layers:
                # Cross attention to image features
                if hasattr(layer, 'cross_attn_image'):
                    ca = layer.cross_attn_image
                    if isinstance(ca, nn.MultiheadAttention):
                        layer.cross_attn_image = LoRAMultiheadAttention(ca, rank=rank, alpha=alpha)
                        lora_params += 2 * rank * ca.embed_dim + 2 * ca.embed_dim * rank

    print(f"[LoRA] Injected adapters with rank={rank}, alpha={alpha}")
    print(f"[LoRA] Added {lora_params:,} trainable LoRA parameters")
    return lora_params


def get_trainable_params_summary(model):
    """Print summary of trainable vs frozen parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*60}")
    print(f"  Model Parameters Summary")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:>15,}")
    print(f"  Trainable parameters: {trainable_params:>15,}")
    print(f"  Frozen parameters:    {frozen_params:>15,}")
    print(f"  Trainable ratio:      {100*trainable_params/total_params:>14.2f}%")
    print(f"{'='*60}\n")
    return trainable_params, frozen_params
