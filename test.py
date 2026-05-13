"""
Enhanced_RRSIS_UOT Evaluation Script

Evaluate trained Enhanced_RRSIS_UOT model on test/val sets.
Computes: mIoU, oIoU (overall IoU), Precision@X thresholds.

Usage:
    python test.py --dataset rrsis_d --data_root /path/to/data --resume ./output/best_model.pth
"""

import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.enhanced_model import Enhanced_RRSIS_UOT


def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Compute segmentation metrics.

    Args:
        pred_masks: [B, 1, H, W] predicted logits
        gt_masks: [B, 1, H, W] ground truth binary masks

    Returns:
        dict with IoU, precision at various thresholds
    """
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

    # Per-sample IoU
    intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Overall IoU (cumulative intersection / cumulative union)
    total_intersection = intersection.sum()
    total_union = union.sum()

    # Precision at thresholds
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = {}
    for t in prec_thresholds:
        precisions[f'P@{t}'] = (iou > t).float().mean().item()

    return {
        'iou': iou,
        'mean_iou': iou.mean().item(),
        'intersection': total_intersection.item(),
        'union': total_union.item(),
        **precisions,
    }


@torch.no_grad()
def evaluate(model, test_loader, device, args):
    """Run full evaluation."""
    model.eval()

    all_ious = []
    total_intersection = 0
    total_union = 0
    prec_counts = {f'P@{t}': 0 for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_samples = 0
    total_time = 0

    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval captions (list of lists)
        if isinstance(captions[0], list):
            # Evaluate on each caption and take best IoU
            best_iou_per_sample = []
            for cap_list in captions:
                ious_for_caps = []
                for cap in cap_list:
                    start = time.time()
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(images[0:1], [cap], masks[0:1])
                    total_time += time.time() - start

                    metrics = compute_metrics(outputs['pred_masks'], masks[0:1])
                    ious_for_caps.append(metrics['iou'].item())

                best_iou_per_sample.append(max(ious_for_caps))

            for iou_val in best_iou_per_sample:
                all_ious.append(iou_val)
                total_samples += 1
                for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    if iou_val > t:
                        prec_counts[f'P@{t}'] += 1
        else:
            # Single caption per sample
            start = time.time()
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images, captions, masks)
            total_time += time.time() - start

            metrics = compute_metrics(outputs['pred_masks'], masks)
            all_ious.extend(metrics['iou'].cpu().numpy().tolist())
            total_intersection += metrics['intersection']
            total_union += metrics['union']
            total_samples += images.size(0)

            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                prec_counts[f'P@{t}'] += (metrics['iou'] > t).sum().item()

        if (batch_idx + 1) % 100 == 0:
            current_miou = np.mean(all_ious)
            print(f"  Progress: {batch_idx+1}/{len(test_loader)}, "
                  f"Current mIoU={current_miou:.4f}")

    # Final metrics
    mIoU = np.mean(all_ious)
    oIoU = total_intersection / (total_union + 1e-6) if total_union > 0 else mIoU

    results = {
        'mIoU': mIoU,
        'oIoU': oIoU,
        'num_samples': total_samples,
        'avg_time': total_time / max(total_samples, 1),
    }
    for key in prec_counts:
        results[key] = prec_counts[key] / total_samples

    return results


def save_predictions(model, test_loader, device, output_dir, args):
    """Save predicted masks as images for visualization."""
    model.eval()
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        if batch_idx >= 50:
            break

        images = images.to(device)
        if isinstance(captions[0], list):
            captions = [cap[0] for cap in captions]

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images, captions)

        pred_probs = torch.sigmoid(outputs['pred_masks'])
        pred_binary = (pred_probs > 0.5).float()

        for i in range(images.size(0)):
            pred_np = (pred_binary[i, 0].cpu().numpy() * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_np)
            pred_img.save(os.path.join(vis_dir, f'{batch_idx}_{i}_pred.png'))

            gt_np = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
            gt_img = Image.fromarray(gt_np)
            gt_img.save(os.path.join(vis_dir, f'{batch_idx}_{i}_gt.png'))

    print(f"  Saved visualizations to {vis_dir}")


def main():
    args = get_args()
    args.eval_only = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Enhanced_RRSIS_UOT Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Device: {device}")
    print(f"  --- Enhancements ---")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss}")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot}")
    print(f"  OHEM Loss: {args.use_ohem_loss}")
    print(f"{'='*60}\n")

    # ====== Build Model ======
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
    )

    # Load trained weights
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
              f"best_iou={ckpt.get('best_iou', '?')}")
    else:
        print("WARNING: No checkpoint provided, evaluating with pretrained SAM3 only!")

    model = model.to(device)
    model.eval()

    # ====== Build Dataset ======
    test_dataset = get_dataset(args, split=args.split, eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Evaluate ======
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, args)

    # ====== Print Results ======
    print(f"\n{'='*60}")
    print(f"  Results on {args.dataset} ({args.split})")
    print(f"{'='*60}")
    print(f"  mIoU:  {results['mIoU']*100:.2f}%")
    print(f"  oIoU:  {results['oIoU']*100:.2f}%")
    print(f"  P@0.5: {results['P@0.5']*100:.2f}%")
    print(f"  P@0.6: {results['P@0.6']*100:.2f}%")
    print(f"  P@0.7: {results['P@0.7']*100:.2f}%")
    print(f"  P@0.8: {results['P@0.8']*100:.2f}%")
    print(f"  P@0.9: {results['P@0.9']*100:.2f}%")
    print(f"  Samples: {results['num_samples']}")
    print(f"  Avg Time: {results['avg_time']*1000:.1f}ms")
    print(f"{'='*60}")

    # ====== Save Visualizations ======
    if args.visualize:
        save_predictions(model, test_loader, device, args.output_dir, args)


if __name__ == '__main__':
    main()
