import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import BallCOCODataset  # uses your heatmap.py logic

# -------- Config --------
root_dir = "../fiba_basketball2/train"
ann_path = os.path.join(root_dir, "annos_train.json")
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# -------- Dataset --------
dataset = BallCOCODataset(
    img_dir=root_dir,
    ann_file=ann_path,
    seq_len=3,
    input_size=(288, 512),
    min_value=0.7
)
loader = DataLoader(dataset, batch_size=1)

# -------- Save one sample --------
for idx, (imgs, hms) in enumerate(loader):
    # Extract center frame from the triple (f1) — channels 3:6
    img_center = imgs[0, 3:6].permute(1, 2, 0).numpy()  # shape: (H, W, 3)
    img_center = (img_center * 255).clip(0, 255).astype(np.uint8)

    # Save center frame
    img_path = os.path.join(output_dir, f"sample_{idx:03d}_center.png")
    cv2.imwrite(img_path, cv2.cvtColor(img_center, cv2.COLOR_RGB2BGR))

    # Extract and visualize heatmap
    print(f"hms keys: {hms.keys()}")
    heatmap = hms[0][0, 0].numpy()
    heatmap_norm = (heatmap * 255).clip(0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(output_dir, f"sample_{idx:03d}_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_colored)

    print(f"[INFO] Saved center frame and heatmap to: {output_dir}")
    break  # save only one for sanity check

