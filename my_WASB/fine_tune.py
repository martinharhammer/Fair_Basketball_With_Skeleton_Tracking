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
NUM_WORKERS = 0
LR_HEAD = 1e-3
EPOCHS = 50
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
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = full_transform

        # Collect unique base names like "triple_001", "triple_002", ...
        all_names = [os.path.basename(img['file_name']) for img in self.coco.dataset['images']]
        self.triples = sorted(list(set(fn.split('_f')[0] for fn in all_names)))

        # Map triple → f1 filename → image ID (for heatmap annotation)
        self.name_to_id = {
            os.path.basename(img['file_name']): img['id']
            for img in self.coco.dataset['images']
        }

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple_id = self.triples[idx]  # e.g., "triple_035"
        frame_names = [f"{triple_id}_f{i}.png" for i in range(self.seq_len)]

        # Load the 3 frames
        seq = [Image.open(os.path.join(self.img_dir, fn)).convert('RGB') for fn in frame_names]
        inp = self.transform(seq)

        # Load heatmap for middle frame (f1)
        mid_name = f"{triple_id}_f1.png"
        mid_id = self.name_to_id[mid_name]
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
            g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 16))
            hm = np.maximum(hm, g)

        print(f"[DEBUG] Sample idx {idx} → frames: {frame_names}")
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

full_dataset = BallCOCODataset(TRAIN_IMG, TRAIN_ANN, SEQ_LEN)
tiny_dataset = torch.utils.data.Subset(full_dataset, list(range(10)))
train_loader = DataLoader(tiny_dataset, batch_size=2, shuffle=False)

#train_loader = DataLoader(BallCOCODataset(TRAIN_IMG, TRAIN_ANN, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#val_loader = DataLoader(BallCOCODataset(VAL_IMG, VAL_ANN, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)



def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE).float()
        out = model(x)[0]
        pred = out[:, out.shape[1] // 2]
        probs = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
        loss = criterion({0: probs.unsqueeze(1)}, {0: y})
        print(f"Target: {y.shape}, Pred: {probs.shape}, Loss: {loss.item():.4f}") 
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
    va = 0.0 #run_epoch(val_loader, False)
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

for x, y in train_loader:
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
