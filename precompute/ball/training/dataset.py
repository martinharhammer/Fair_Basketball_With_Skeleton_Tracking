import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from heatmap import gen_heatmap
from preprocessing import ResizeWithEqualScale, SeqTransformCompose

class BallCOCODataset(Dataset):
    def __init__(self, img_dir, ann_file, seq_len=3, input_size=(288, 512), min_value=0.7):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.inp_h, self.inp_w = input_size
        self.min_value = min_value

        self.coco = COCO(ann_file)
        self.transform = SeqTransformCompose(frame_transform=ResizeWithEqualScale(height=self.inp_h, width=self.inp_w))

        all_filenames = [os.path.basename(img['file_name']) for img in self.coco.dataset['images']]
        self.triples = sorted(list(set(fn.split('_f')[0] for fn in all_filenames)))

        self.name_to_id = {
            os.path.basename(img['file_name']): img['id']
            for img in self.coco.dataset['images']
        }

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple_id = self.triples[idx]
        frame_names = [f"{triple_id}_f{i}.png" for i in range(self.seq_len)]
        frame_paths = [os.path.join(self.img_dir, fn) for fn in frame_names]

        if not all(os.path.exists(p) for p in frame_paths):
            print(f"[WARN] Skipping incomplete triple: {triple_id}")
            return self.__getitem__((idx + 1) % len(self))

        seq = [Image.open(p).convert('RGB') for p in frame_paths]
        inp = self.transform(seq)  # shape: (9, H, W)

        mid_name = f"{triple_id}_f1.png"
        mid_id = self.name_to_id[mid_name]
        info = self.coco.loadImgs(mid_id)[0]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=mid_id))

        if not anns:
            print(f"[WARN] No annotations for {mid_name}")
            return self.__getitem__((idx + 1) % len(self))

        bbox = anns[0]['bbox']
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        r = np.sqrt(w * h) * 0.15

        orig_w, orig_h = info['width'], info['height']
        if self.inp_h / self.inp_w >= orig_h / orig_w:
            new_w = self.inp_w
            new_h = int(orig_h * (self.inp_w / orig_w))
        else:
            new_h = self.inp_h
            new_w = int(orig_w * (self.inp_h / orig_h))

        pad_x = (self.inp_w - new_w) // 2
        pad_y = (self.inp_h - new_h) // 2
        cx = cx * (new_w / orig_w) + pad_x
        cy = cy * (new_h / orig_h) + pad_y

        if not (0 <= cx < self.inp_w) or not (0 <= cy < self.inp_h):
            print(f"[WARN] Center outside bounds in {mid_name}: ({cx:.1f}, {cy:.1f})")
            return self.__getitem__((idx + 1) % len(self))

        heatmap = gen_heatmap(
            wh=(self.inp_w, self.inp_h),
            cxy=(cx, cy),
            r=r,
            min_value=self.min_value,
            data_type=np.float32
        )

        return inp, {0: torch.tensor(heatmap).unsqueeze(0)}
