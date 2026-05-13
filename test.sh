#!/bin/bash
# ====== Enhanced_RRSIS_UOT Evaluation Script ======
#
# Usage:
#   bash test.sh <dataset_name> <split> <checkpoint_path>
#   e.g., bash test.sh rrsis_d test ./output/rrsis_d_enhanced_uot/best_model.pth

DATASET=${1:-rrsis_d}
SPLIT=${2:-test}
CKPT=${3:-./output/${DATASET}_enhanced_uot/best_model.pth}
DATA_ROOT=${4:-./data}

echo "============================================="
echo "  Enhanced_RRSIS_UOT Evaluation"
echo "  Dataset: ${DATASET}"
echo "  Split: ${SPLIT}"
echo "  Checkpoint: ${CKPT}"
echo "============================================="

python test.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --split ${SPLIT} \
    --resume ${CKPT} \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --eval_only \
    --visualize \
    --num_workers 4
