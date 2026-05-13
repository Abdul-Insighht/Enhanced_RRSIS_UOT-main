# Enhanced_RRSIS_UOT: Enhanced Referring Remote Sensing Image Segmentation with Unbalanced Optimal Transport

**Enhanced_RRSIS_UOT** extends [RRSIS_SAM3](../RRSIS_SAM3/) with **4 novel techniques** for improved performance on referring remote sensing image segmentation.

## What's New (Over RRSIS_SAM3)

| Enhancement | Module | Description |
|-------------|--------|-------------|
| 🟢 **Text-Guided Dynamic LoRA** | `lib/dynamic_lora.py` | Text-conditioned vision adapter weights — vision encoder adapts per-caption |
| 🟢 **Contrastive Loss (InfoNCE)** | `lib/contrastive_loss.py` | Auxiliary loss aligning masked visual features with text features |
| 🟢 **Multi-Scale OT Alignment** | `lib/multiscale_ot_alignment.py` | Scale-aware OT alignment across all FPN levels with gated residual |
| 🟢 **OHEM + Focal + Boundary Loss** | `lib/ohem_loss.py` | Hard pixel mining + focal weighting + boundary supervision |
| 🟢 **Pipeline B (Semantic Prompter)**| `lib/pipeline_b_vit_bert.py` | Optimized Feature-Reuse Prompter — auto-generates visual prompts from SAM3 features |

## Architecture

```text
Image (504×504) + Text Caption
    │
    ├── [Part A: Main SAM3 Pipeline]
    │    ├── SAM3 VL Backbone (ViT + Text Encoder, frozen)
    │    │       + Text-Guided Dynamic LoRA (text-conditioned adapters) ← NEW
    │    │
    │    ├── Multi-Scale OT Alignment (per-FPN-level Sinkhorn matching) ← ENHANCED
    │
    ├── [Part B: Semantic Referring Prompter] ← NEW
    │    ├── Feature-Reuse Cross-Attention (uses SAM3 features)
    │    └── Prompter Module → Generates Dense & Sparse Prompts
    │
    ├── Transformer Encoder (Fusion of Part A features + Part B Prompts)
    │
    ├── DETR Decoder (object detection, fine-tuned)
    │
    ├── Segmentation Head (mask prediction, fine-tuned)
    │
    └── Loss Computation:
            OHEM + FocalDice + Boundary Loss ← ENHANCED
            + Contrastive InfoNCE Loss (auxiliary) ← NEW
```

## Key Differences from RRSIS_SAM3

| Feature | RRSIS_SAM3 | Enhanced_RRSIS_UOT |
|---------|------------|---------------------|
| LoRA Type | Static (same weights for all inputs) | **Dynamic** (text-conditioned per-caption) |
| OT Alignment | Single-scale, one pass | **Multi-scale** across all FPN levels |
| Loss Function | Dice + BCE | **OHEM + FocalDice + Boundary + Contrastive** |
| Visual Prompts| None (relies on global features) | **Pipeline B** (Auto-generates visual prompts) |
| Vision-Language Bond | Fusion encoder only | **Early** (LoRA) + **Mid** (OT) + **Late** (Contrastive) |

## Supported Datasets

| Dataset | Train | Val | Test | Image Size | Categories |
|---------|-------|-----|------|------------|------------|
| **RRSIS-D** | 12,181 | 1,740 | 3,481 | 800×800 | 20 |
| **RRSIS-HR** | 2,118 | 268 | 264 | 1024×1024 | 7 |
| **RefSegRS** | 2,172 | 413 | 1,817 | 512×512 | — |

## Training

### Full Enhanced Model (All 4 Techniques)
```bash
bash fine.sh rrsis_d ./data
```

### Ablation Studies
```bash
# Baseline (equivalent to RRSIS_SAM3):
bash fine.sh rrsis_d ./data --no_dynamic_lora --no_contrastive_loss --no_multiscale_ot --no_ohem_loss

# Only Dynamic LoRA:
bash fine.sh rrsis_d ./data --no_contrastive_loss --no_multiscale_ot --no_ohem_loss

# Only OHEM Loss:
bash fine.sh rrsis_d ./data --no_dynamic_lora --no_contrastive_loss --no_multiscale_ot

# Dynamic LoRA + OHEM (best combo):
bash fine.sh rrsis_d ./data --no_contrastive_loss --no_multiscale_ot
```

### Key Training Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_dynamic_lora` | True | Enable text-guided dynamic LoRA |
| `--use_contrastive_loss` | True | Enable InfoNCE contrastive loss |
| `--use_multiscale_ot` | True | Enable multi-scale OT alignment |
| `--use_ohem_loss` | True | Enable OHEM + Focal + Boundary loss |
| `--contrastive_weight` | 0.1 | Weight for contrastive loss |
| `--ohem_hard_ratio` | 0.3 | Fraction of hard pixels for OHEM |
| `--num_ot_scales` | 3 | Number of FPN scales for OT |
| `--focal_gamma` | 2.0 | Focal loss focusing parameter |

## Evaluation

```bash
python test.py --dataset rrsis_d --split test --resume ./output/rrsis_d_enhanced_uot/best_model.pth
```

### Output Metrics
- **mIoU**: Mean Intersection over Union
- **oIoU**: Overall IoU
- **P@0.5 - P@0.9**: Precision at IoU thresholds

## Project Structure
```
Enhanced_RRSIS_UOT/
├── sam3/                           # SAM3 core (Meta's implementation)
├── lib/
│   ├── enhanced_model.py           # ★ Enhanced model (main entry point)
│   ├── dynamic_lora.py             # ★ Text-Guided Dynamic LoRA
│   ├── contrastive_loss.py         # ★ InfoNCE Contrastive Loss
│   ├── multiscale_ot_alignment.py  # ★ Multi-Scale OT Alignment
│   ├── ohem_loss.py                # ★ OHEM + Focal + Boundary Loss
│   ├── pipeline_b_vit_bert.py      # ★ Pipeline B (Semantic Prompter)
│   ├── rrsis_sam3_model.py         # Base model (from RRSIS_SAM3)
│   ├── rs_adapters.py              # Static LoRA adapters (fallback)
│   ├── ot_feature_alignment.py     # Single-scale OT (fallback)
│   └── ot_loss.py                  # Standard Dice+BCE (fallback)
├── data/                           # Dataset loaders
├── refer/                          # REFER API
├── loss/                           # Legacy loss functions
├── configs/
│   └── enhanced_rrsis_uot.yaml     # Full configuration
├── train.py                        # Training script
├── test.py                         # Evaluation script
├── args.py                         # CLI arguments
├── fine.sh                         # Training launcher
└── test.sh                         # Evaluation launcher
```

## Citation
```bibtex
@article{enhanced_rrsis_uot_2026,
    title={Enhanced RRSIS-UOT: Enhanced Referring Remote Sensing Image Segmentation
           with Unbalanced Optimal Transport},
    year={2026}
}
```
