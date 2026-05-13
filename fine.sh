#!/bin/bash
# ====== Enhanced_RRSIS_UOT Training Script ======
# Optimized for Kaggle T4/P100 (16GB VRAM)
# Includes all 4 enhancements: Dynamic LoRA, Contrastive Loss,
# Multi-Scale OT, OHEM Loss
#
# Usage:
#   bash fine.sh <dataset_name>
#   e.g., bash fine.sh rrsis_d
#         bash fine.sh rrsis_hr
#         bash fine.sh refsegrs
#
# Ablation (disable specific techniques):
#   bash fine.sh rrsis_d ./data --no_dynamic_lora
#   bash fine.sh rrsis_d ./data --no_contrastive_loss --no_ohem_loss

DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
OUTPUT_DIR="./output/${DATASET}_enhanced_uot"
EXTRA_ARGS="${@:3}"

echo "============================================="
echo "  Enhanced_RRSIS_UOT Training"
echo "  Dataset: ${DATASET}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Extra Args: ${EXTRA_ARGS}"
echo "============================================="

python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    --contrastive_weight 0.1 \
    --ohem_hard_ratio 0.3 \
    --ot_reg 0.1 \
    --ot_num_iter 10 \
    --num_ot_scales 3 \
    --focal_gamma 2.0 \
    ${EXTRA_ARGS}
