import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from hrnet import HRNet
from omegaconf import OmegaConf
from preprocessing import ResizeWithEqualScale, SeqTransformCompose
from losses.quality_focal_loss import QualityFocalLoss

# CONFIG
DATA_ROOT = '/home/ubuntu/my_WASB/striped_ball_dataset2'
TRAIN_IMG = os.path.join(DATA_ROOT, 'images/train')
TRAIN_ANN = os.path.join(DATA_ROOT, 'annotations/instances_train.json')
VAL_IMG = os.path.join(DATA_ROOT, 'images/val')
VAL_ANN = os.path.join(DATA_ROOT, 'annotations/instances_val.json')
PRETRAINED = '/home/ubuntu/my_WASB/wasb_basketball_best.pth.tar'
CONFIG_YAML = 'config_hrnet.yaml'
OUTPUT_DIR = 'finetune_outputs'
BATCH_SIZE = 4
NUM_WORKERS = 2
LR_HEAD = 1e-3
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
cfg = OmegaConf.load(CONFIG_YAML)
SEQ_LEN = cfg.frames_in
INP_H, INP_W = cfg.inp_height, cfg.inp_width
full_transform = SeqTransformCompose(frame_transform=ResizeWithEqualScale(INP_H, INP_W))

# DATASET
class BallCOCODataset(Dataset):
    def __init__(self, img_dir, ann_file, seq_len):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        imgs = sorted(self.coco.imgs.values(), key=lambda x: os.path.basename(x['file_name']))
        self.ids = [img['id'] for img in imgs]
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = full_transform

    def __len__(self):
        return len(self.ids) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = []
        for i in range(idx, idx + self.seq_len):
            info = self.coco.loadImgs(self.ids[i])[0]
            fname = os.path.basename(info['file_name'].replace('\\', '/'))
            path = os.path.join(self.img_dir, fname)
            seq.append(Image.open(path).convert('RGB'))
        inp = self.transform(seq)
        mid_id = self.ids[idx + self.seq_len // 2]
        info_mid = self.coco.loadImgs(mid_id)[0]
        anns_mid = self.coco.loadAnns(self.coco.getAnnIds(imgIds=mid_id))
        h, w = inp.shape[-2:]
        hm = np.zeros((h, w), np.float32)
        for ann in anns_mid:
            x, y, wb, hb = ann['bbox']
            cx, cy = x + wb / 2, y + hb / 2
            sx, sy = w / info_mid['width'], h / info_mid['height']
            cx, cy = cx * sx, cy * sy
            yy, xx = np.ogrid[:h, :w]
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 16))
            hm = np.maximum(hm, g)
        return inp, torch.from_numpy(hm).unsqueeze(0)

# MODEL
model = HRNet(cfg)
model.init_weights(pretrained=PRETRAINED)
model.to(DEVICE)

for name, param in model.named_parameters():
    param.requires_grad = 'final_layers' in name or 'stage4' in name

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = QualityFocalLoss(beta=2, auto_weight=False, scales=[0])

train_loader = DataLoader(BallCOCODataset(TRAIN_IMG, TRAIN_ANN, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(BallCOCODataset(VAL_IMG, VAL_ANN, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE).float()
        out = model(x)[0]
        print(f"Input shape: {x.shape} | Output shape: {out.shape}")
        pred = out[:, out.shape[1] // 2]
        probs = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
        loss = criterion({0: probs.unsqueeze(1)}, {0: y})
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

# TRAIN LOOP
best_val = float('inf')
for ep in range(EPOCHS):
    if ep == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"[DEBUG] Trainable layer: {name}")
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    scheduler.step()
    print(f"Epoch {ep + 1}/{EPOCHS}  train={tr:.4f}  val={va:.4f}")
    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_head_only.pth'))

print(f"Done. Best val loss: {best_val:.4f}")

import matplotlib.pyplot as plt

model.eval()
out_dir = os.path.join(OUTPUT_DIR, "sanity_check_heatmaps")
os.makedirs(out_dir, exist_ok=True)

for x, y in val_loader:
    x, y = x.to(DEVICE), y.to(DEVICE).float()
    with torch.no_grad():
        out = model(x)[0]  # Tensor: (B, T, H, W)
        pred = out[:, out.shape[1] // 2].sigmoid().cpu().numpy()  # (B, H, W)
        gt = y.cpu().numpy()[:, 0]  # (B, 1, H, W) → (B, H, W)
    for i in range(min(4, x.size(0))):
        # Convert input back to PIL-compatible RGB image
        inp_seq = x[i].cpu().numpy()  # Shape: (3*N, H, W)
        mid = SEQ_LEN // 2
        img_rgb = inp_seq[mid*3:(mid+1)*3]  # Middle frame only
        img_rgb = np.transpose(img_rgb, (1, 2, 0))  # CHW → HWC
        img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_rgb)
        axs[0].set_title("Input Frame")
        axs[1].imshow(gt[i], cmap='hot')
        axs[1].set_title("GT Heatmap")
        axs[2].imshow(pred[i], cmap='hot')
        axs[2].set_title("Predicted Heatmap")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        fname = os.path.join(out_dir, f"sample_{i:02d}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    break  # only one batch
print(f"✅ Sanity check heatmaps saved to {out_dir}")
