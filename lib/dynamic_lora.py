"""
Text-Guided Dynamic LoRA for Enhanced_RRSIS_UOT.

Instead of static LoRA adapters (same ΔW for every input), Dynamic LoRA
generates text-conditioned LoRA matrices so the vision encoder becomes
text-aware from the very first layer.

Key idea:
    - A small MLP (HyperNetwork) takes the pooled text embedding
    - It generates LoRA A and B matrices conditioned on the text
    - Each image sees a DIFFERENT set of adapter weights depending
      on what the referring expression asks for

This is our novel contribution: text-guided vision adaptation.

Reference:
    Inspired by HyperNetworks (Ha et al., 2016) and
    LoRA (Hu et al., 2021) — combined for cross-modal adaptation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedLoRA(nn.Module):
    """
    HyperNetwork that generates LoRA A and B matrices from text features.

    Given pooled text features, produces low-rank adapter matrices that
    are specific to the referring expression, allowing the vision encoder
    to attend to text-relevant visual patterns from the start.

    Args:
        text_dim: Dimension of input text features.
        in_features: Input dimension of the target linear layer.
        out_features: Output dimension of the target linear layer.
        rank: LoRA rank (low-rank bottleneck dimension).
        alpha: LoRA scaling factor.
    """

    def __init__(self, text_dim, in_features, out_features, rank=16, alpha=32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = in_features
        self.out_features = out_features

        # HyperNetwork: text → LoRA_A (rank × in_features)
        # Use a 2-layer MLP with bottleneck for efficiency
        hyper_bottleneck = min(256, text_dim // 2)
        self.hyper_A = nn.Sequential(
            nn.Linear(text_dim, hyper_bottleneck),
            nn.GELU(),
            nn.Linear(hyper_bottleneck, rank * in_features),
        )

        # HyperNetwork: text → LoRA_B (out_features × rank)
        self.hyper_B = nn.Sequential(
            nn.Linear(text_dim, hyper_bottleneck),
            nn.GELU(),
            nn.Linear(hyper_bottleneck, out_features * rank),
        )

        # Initialize last layer near zero so ΔW starts small
        nn.init.zeros_(self.hyper_A[-1].weight)
        nn.init.zeros_(self.hyper_A[-1].bias)
        nn.init.zeros_(self.hyper_B[-1].weight)
        nn.init.zeros_(self.hyper_B[-1].bias)

    def forward(self, text_feat):
        """
        Generate LoRA matrices from text features.

        Args:
            text_feat: (B, text_dim) pooled text embedding.

        Returns:
            lora_A: (B, rank, in_features)
            lora_B: (B, out_features, rank)
        """
        B = text_feat.shape[0]
        lora_A = self.hyper_A(text_feat).view(B, self.rank, self.in_features)
        lora_B = self.hyper_B(text_feat).view(B, self.out_features, self.rank)
        return lora_A, lora_B


class DynamicLoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a text-conditioned Dynamic LoRA adapter.

    The original linear is frozen. During forward, if text features are available,
    dynamic LoRA matrices are generated; otherwise falls back to static LoRA.

    Output = Original(x) + scaling * (x @ A^T @ B^T)
    where A, B are generated from text features.

    Args:
        original_linear: The frozen nn.Linear layer to wrap.
        text_dim: Dimension of text conditioning features.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
    """

    def __init__(self, original_linear, text_dim=256, rank=16, alpha=32.0):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Dynamic LoRA generator
        self.dynamic_lora = TextConditionedLoRA(
            text_dim=text_dim,
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
        )

        # Static fallback LoRA (used when no text is available, e.g. init)
        self.static_lora_A = nn.Parameter(torch.zeros(rank, original_linear.in_features))
        self.static_lora_B = nn.Parameter(torch.zeros(original_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.static_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.static_lora_B)

        # Cached text features for current forward pass
        self._cached_text_feat = None

    def set_text_conditioning(self, text_feat):
        """
        Cache text features for the current forward pass.

        Args:
            text_feat: (B, text_dim) pooled text embedding.
        """
        self._cached_text_feat = text_feat

    def clear_text_conditioning(self):
        """Clear cached text features after forward pass."""
        self._cached_text_feat = None

    def forward(self, x):
        """
        Forward pass with dynamic or static LoRA.

        Args:
            x: (*, in_features) input tensor.

        Returns:
            (*, out_features) output tensor.
        """
        base_out = self.original_linear(x)

        if self._cached_text_feat is not None:
            # Dynamic LoRA: text-conditioned adaptation
            lora_A, lora_B = self.dynamic_lora(self._cached_text_feat)
            B = lora_A.shape[0]

            # x shape: could be (B, N, D) or (B*N, D)
            if x.dim() == 3:
                # (B, N, D) @ (B, D, rank) → (B, N, rank)
                delta = torch.bmm(x, lora_A.transpose(1, 2))
                # (B, N, rank) @ (B, rank, out_D) → (B, N, out_D)
                delta = torch.bmm(delta, lora_B.transpose(1, 2))
            elif x.dim() == 2 and x.shape[0] == B:
                # (B, D) → (B, 1, D) for bmm
                delta = torch.bmm(x.unsqueeze(1), lora_A.transpose(1, 2))
                delta = torch.bmm(delta, lora_B.transpose(1, 2)).squeeze(1)
            else:
                # Fallback to static LoRA
                delta = (x @ self.static_lora_A.T @ self.static_lora_B.T)

            return base_out + self.scaling * delta
        else:
            # Static LoRA fallback
            delta = (x @ self.static_lora_A.T @ self.static_lora_B.T)
            return base_out + self.scaling * delta


class DynamicLoRAManager:
    """
    Manages text conditioning across all DynamicLoRALinear layers in a model.

    Call set_text_conditioning() before forward pass to propagate
    text features to all dynamic LoRA layers, and clear_text_conditioning()
    after the pass.
    """

    def __init__(self, model):
        self.dynamic_lora_layers = []
        for module in model.modules():
            if isinstance(module, DynamicLoRALinear):
                self.dynamic_lora_layers.append(module)

    def set_text_conditioning(self, text_feat):
        """Propagate text features to all dynamic LoRA layers."""
        for layer in self.dynamic_lora_layers:
            layer.set_text_conditioning(text_feat)

    def clear_text_conditioning(self):
        """Clear text features from all dynamic LoRA layers."""
        for layer in self.dynamic_lora_layers:
            layer.clear_text_conditioning()


def inject_dynamic_lora_adapters(model, text_dim=256, rank=16, alpha=32.0):
    """
    Inject Dynamic LoRA adapters into SAM3's ViT backbone.

    Replaces attention QKV and projection layers with text-conditioned
    dynamic LoRA wrappers.

    Args:
        model: The SAM3 model (Sam3Image).
        text_dim: Dimension of text features for conditioning.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Tuple of (num_params_added, DynamicLoRAManager).
    """
    lora_params = 0

    backbone = model.backbone
    if hasattr(backbone, 'vision_backbone'):
        vit = backbone.vision_backbone
        if hasattr(vit, 'trunk'):
            trunk = vit.trunk
            if hasattr(trunk, 'blocks'):
                for i, block in enumerate(trunk.blocks):
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        # Dynamic LoRA on qkv projection
                        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
                            original = attn.qkv
                            attn.qkv = DynamicLoRALinear(
                                original, text_dim=text_dim,
                                rank=rank, alpha=alpha
                            )
                            lora_params += sum(
                                p.numel() for p in attn.qkv.parameters()
                                if p.requires_grad
                            )
                        # Dynamic LoRA on output projection
                        if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
                            original = attn.proj
                            attn.proj = DynamicLoRALinear(
                                original, text_dim=text_dim,
                                rank=rank, alpha=alpha
                            )
                            lora_params += sum(
                                p.numel() for p in attn.proj.parameters()
                                if p.requires_grad
                            )

    manager = DynamicLoRAManager(model)

    print(f"[DynamicLoRA] Injected text-conditioned adapters (text_dim={text_dim}, rank={rank}, alpha={alpha})")
    print(f"[DynamicLoRA] Added {lora_params:,} trainable dynamic LoRA parameters")

    return lora_params, manager
