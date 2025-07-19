import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from hrnet import HRNet
from preprocessing import ResizeWithEqualScale, SeqTransformCompose

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_ROOT = '/home/ubuntu/my_WASB/striped3'
TRAIN_IMG = os.path.join(DATA_ROOT, 'images/train')
TRAIN_ANN = os.path.join(DATA_ROOT, 'annotations/instances_default.json')
PRETRAINED = '/home/ubuntu/my_WASB/wasb_basketball_best.pth.tar'
CONFIG_YAML = 'config_hrnet.yaml'
OUTPUT_DIR = 'finetune_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = OmegaConf.load(CONFIG_YAML)
SEQ_LEN = cfg.frames_in
INP_H, INP_W = cfg.inp_height, cfg.inp_width
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# TRANSFORM + DATASET
# ----------------------------
resize        = ResizeWithEqualScale(height=cfg['inp_height'],
                                     width= cfg['inp_width'])
full_transform = SeqTransformCompose(frame_transform=resize)

class BallCOCODataset(Dataset):
    def __init__(self, img_dir, ann_file, seq_len):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = full_transform

        all_names = [os.path.basename(img['file_name']) for img in self.coco.dataset['images']]
        self.triples = sorted(list(set(fn.split('_f')[0] for fn in all_names)))

        self.name_to_id = {
            os.path.basename(img['file_name']): img['id']
            for img in self.coco.dataset['images']
        }

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple_id = self.triples[idx]
        frame_names = [f"{triple_id}_f{i}.png" for i in range(self.seq_len)]
        seq = [Image.open(os.path.join(self.img_dir, fn)).convert('RGB') for fn in frame_names]
        inp = self.transform(seq)

        mid_name = f"{triple_id}_f1.png"
        mid_id = self.name_to_id[mid_name]
        info_mid = self.coco.loadImgs(mid_id)[0]
        print(f"[DEBUG] Loading image: {info_mid['file_name']} (id: {mid_id})")
        anns_mid = self.coco.loadAnns(self.coco.getAnnIds(imgIds=mid_id))
        if anns_mid:
            for i, ann in enumerate(anns_mid):
                print(f"[DEBUG] Ann #{i}: bbox={ann.get('bbox')}, category_id={ann.get('category_id')}")
        else:
            print("[DEBUG] No annotations found.")

        h, w = inp.shape[-2:]
        hm = np.zeros((h, w), np.float32)
        cx, cy = -1, -1

        if anns_mid and 'bbox' in anns_mid[0] and anns_mid[0]['bbox'][2] > 0 and anns_mid[0]['bbox'][3] > 0:
            ann = anns_mid[0]
            x, y, wb, hb = ann['bbox']
            cx, cy = x + wb / 2, y + hb / 2
            orig_w, orig_h = info_mid['width'], info_mid['height']

            if INP_H / INP_W >= orig_h / orig_w:
                new_w = INP_W
                new_h = int(orig_h * (INP_W / orig_w))
            else:
                new_h = INP_H
                new_w = int(orig_w * (INP_H / orig_h))
            pad_x = (INP_W - new_w) // 2
            pad_y = (INP_H - new_h) // 2
            cx = cx * (new_w / orig_w) + pad_x
            cy = cy * (new_h / orig_h) + pad_y

            if not (0 <= cx < INP_W) or not (0 <= cy < INP_H):
                print(f"[WARNING] Skipping: off-image center ({cx:.2f}, {cy:.2f})")
                return self.__getitem__((idx + 1) % len(self))

            yy, xx = np.ogrid[:h, :w]
            sigma = 0.7  # Smaller = sharper peak
            g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            hm = np.maximum(hm, g)
        else:
            print(f"[WARNING] Skipping: Invalid bbox in {mid_name}")
            return self.__getitem__((idx + 1) % len(self))

        return inp, torch.from_numpy(hm).unsqueeze(0), torch.tensor([cx, cy])

# ----------------------------
# MODEL SETUP
# ----------------------------
model = HRNet(cfg)
model.init_weights(pretrained=PRETRAINED)
for name, module in model.named_modules():
    if 'final_layers' in name:
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.normal_(module.weight, std=0.001)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
model.to(DEVICE)

for name, param in model.named_parameters():
    param.requires_grad = any(x in name for x in ['stage4', 'final_layers'])

pos_weight = torch.tensor([1.0]).to(DEVICE)  # or try 50.0 if needed
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ----------------------------
# TRAINING ON triple_001 ONLY
# ----------------------------
dataset = BallCOCODataset(TRAIN_IMG, TRAIN_ANN, SEQ_LEN)
subset = Subset(dataset, [0])
train_loader = DataLoader(subset, batch_size=1, shuffle=False)

EPOCHS = 50
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y, _ in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float()

        pred_dict = model(x)
        pred_mid = pred_dict[0][:, pred_dict[0].shape[1] // 2].unsqueeze(1)
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(pred_mid)
            print(f"[DEBUG] Sigmoid pred: min={pred_sigmoid.min().item():.4f}, max={pred_sigmoid.max().item():.4f}, mean={pred_sigmoid.mean().item():.4f}")
            print(f"[DEBUG] Target     : min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}")


        if torch.isnan(pred_mid).any() or torch.isnan(y).any():
            print("[ERROR] NaN detected in input or output!")
            continue
        
        loss = criterion(pred_mid, y)

        print(f"[Epoch {epoch+1}] pred min={pred_mid.min().item():.4f}, max={pred_mid.max().item():.4f}, "
              f"target min={y.min().item():.4f}, max={y.max().item():.4f}, loss={loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        print(f"Grad norm: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.4f}")
        optimizer.step()
        total_loss += loss.item()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_head_only.pth"))
            print(f"[Epoch {epoch+1}] ✅ Saved new best model with loss {best_loss:.6f}")

    scheduler.step()
    print(f"[Epoch {epoch+1}/{EPOCHS}] Avg Loss: {total_loss:.6f}")
