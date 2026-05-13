"""
Dataset loaders for RRSIS_SAM3.

Supports three datasets:
    1. RRSIS-D: 17402 triplets, 20 categories, 800×800
    2. RRSIS-HR: 2650 triplets, 7 categories, 1024×1024
    3. RefSegRS: 4402 triplets, 512×512

All images are resized to 504×504 (divisible by 14 for SAM3's ViT patch size).
No BEiT-3 tokenizer needed — SAM3 handles tokenization internally.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

from refer.refer import REFER


# ============================================================
# RRSIS-D Dataset (uses REFER API)
# ============================================================
class RRSISDDataset(data.Dataset):
    """
    RRSIS-D dataset loader.

    Built on the DIOR-RSVG dataset with 20 object categories.
    12181 train / 1740 val / 3481 test triplets.
    """

    def __init__(self, args, split='train', eval_mode=False):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size

        # Load REFER annotations
        self.refer = REFER(args.data_root, 'rrsis_d', args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        # Build captions
        self.captions = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            caption_for_ref = [el['raw'] for el in ref['sentences']]
            self.captions.append(caption_for_ref)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # Random augmentation for training
        if split == 'train':
            num_to_mask = int(len(ref_ids) * 0.2)
            self.images_to_mask = set(random.sample(ref_ids, num_to_mask))
        else:
            self.images_to_mask = set()

        print(f"[RRSIS-D] Loaded {len(self.ref_ids)} samples for {split}")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # Load image
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert('RGB')

        # Random occlusion augmentation
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = self._add_random_boxes(img)

        # Load mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1

        # Resize image and mask
        img = self.img_transform(img)
        mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]  # Return all captions for eval
        else:
            caption = random.choice(self.captions[index])

        return img, mask, caption

    @staticmethod
    def _add_random_boxes(img, min_num=20, max_num=60, box_size=32):
        img_np = np.asarray(img).copy()
        img_size = img_np.shape[1]
        num = random.randint(min_num, max_num)
        for _ in range(num):
            y = random.randint(0, img_size - box_size)
            x = random.randint(0, img_size - box_size)
            img_np[y:y+box_size, x:x+box_size] = 0
        return Image.fromarray(img_np, 'RGB')


# ============================================================
# RRSIS-HR Dataset (uses REFER API)
# ============================================================
class RRSISHRDataset(data.Dataset):
    """
    RRSIS-HR dataset loader.

    Very high resolution RS images with longer language expressions.
    2118 train / 268 val / 264 test triplets.
    7 object categories, 1024×1024 images.
    """

    def __init__(self, args, split='train', eval_mode=False):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size

        # Load REFER annotations
        self.refer = REFER(args.data_root, 'rrsis_hr', args.splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)
        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        # Build captions
        self.captions = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            caption_for_ref = [el['raw'] for el in ref['sentences']]
            self.captions.append(caption_for_ref)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        print(f"[RRSIS-HR] Loaded {len(self.ref_ids)} samples for {split}")

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        # Load image
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert('RGB')

        # Load mask
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1

        # Resize
        img = self.img_transform(img)
        mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]
        else:
            caption = random.choice(self.captions[index])

        return img, mask, caption


# ============================================================
# RefSegRS Dataset (file-based, not REFER API)
# ============================================================
class RefSegRSDataset(data.Dataset):
    """
    RefSegRS dataset loader.

    First remote sensing referring segmentation dataset.
    2172 train / 413 val / 1817 test, 512×512 images.
    """

    def __init__(self, args, split='train', eval_mode=False, data_root=None):
        self.split = split
        self.eval_mode = eval_mode
        self.image_size = args.image_size
        self.data_root = data_root or os.path.join(args.data_root, 'RefSegRS')

        # Load data from text files
        self.imgs, self.labels, self.sentences = self._build_batches(split)

        # Build caption lists
        self.captions = []
        for sent in self.sentences:
            self.captions.append([sent.strip()])

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        print(f"[RefSegRS] Loaded {len(self.imgs)} samples for {split}")

    def _build_batches(self, split):
        """Load image/mask paths and captions from text files."""
        im_dir = os.path.join(self.data_root, 'images')
        seg_dir = os.path.join(self.data_root, 'masks')

        split_files = {
            'train': 'output_phrase_train.txt',
            'val': 'output_phrase_val.txt',
            'test': 'output_phrase_test.txt',
        }
        set_file = os.path.join(self.data_root, split_files[split])

        all_imgs, all_labels, all_sentences = [], [], []
        with open(set_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                img_name = parts[0]
                sentence = ' '.join(parts[1:])

                all_imgs.append(os.path.join(im_dir, img_name + '.tif'))
                all_labels.append(os.path.join(seg_dir, img_name + '.tif'))
                all_sentences.append(sentence)

        return all_imgs, all_labels, all_sentences

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.imgs[index]).convert('RGB')

        # Load mask
        label_mask = cv2.imread(self.labels[index], 2)
        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1

        # Resize
        img = self.img_transform(img)
        mask = torch.from_numpy(annot).unsqueeze(0).float()
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Select caption
        if self.eval_mode:
            caption = self.captions[index]
        else:
            caption = self.captions[index][0]

        return img, mask, caption


# ============================================================
# Dataset Factory
# ============================================================
def get_dataset(args, split='train', eval_mode=False):
    """
    Get the appropriate dataset based on args.dataset.

    Args:
        args: Parsed arguments with dataset name and paths
        split: 'train', 'val', or 'test'
        eval_mode: If True, return all captions per sample

    Returns:
        Dataset instance
    """
    dataset_map = {
        'rrsis_d': RRSISDDataset,
        'rrsis_hr': RRSISHRDataset,
        'refsegrs': RefSegRSDataset,
    }

    if args.dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {list(dataset_map.keys())}")

    dataset_class = dataset_map[args.dataset]
    return dataset_class(args, split=split, eval_mode=eval_mode)


def collate_fn(batch):
    """
    Custom collate function for RRSIS datasets.

    Handles variable-length captions (strings) alongside tensor data.
    """
    images, masks, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    # captions remain as list of strings (or list of lists for eval)
    return images, masks, list(captions)
