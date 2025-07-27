import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from dataset import BallCOCODataset
from hrnet import HRNet
from runner_utils import train_epoch, test_epoch
from utils import save_checkpoint, count_params
from omegaconf import OmegaConf
from heatmapLossWrapper import HeatmapLoss

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = OmegaConf.load('config_hrnet.yaml')

IMG_DIR_TRAIN   = '../fiba_basketball2/train'
ANN_FILE_TRAIN  = '../fiba_basketball2/train/annos_train.json'
IMG_DIR_VAL     = '../fiba_basketball2/val'
ANN_FILE_VAL    = '../fiba_basketball2/val/annos_val.json'
PRETRAINED_PATH = '../wasb_basketball_best.pth.tar'
OUTPUT_DIR      = 'finetune_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# HYPERPARAMETERS
# ----------------------------
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS      = 50
BATCH_SIZE  = 4
LR          = 5e-4
POS_WEIGHT  = torch.tensor([50.0]).to(DEVICE)

# ----------------------------
# DATALOADERS
# ----------------------------
train_dataset = BallCOCODataset(
    img_dir=IMG_DIR_TRAIN,
    ann_file=ANN_FILE_TRAIN,
    input_size=(CONFIG['inp_height'], CONFIG['inp_width'])
)

val_dataset = BallCOCODataset(
    img_dir=IMG_DIR_VAL,
    ann_file=ANN_FILE_VAL,
    input_size=(CONFIG['inp_height'], CONFIG['inp_width'])
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = HRNet(CONFIG)
ckpt = torch.load(PRETRAINED_PATH, map_location=DEVICE)

# Filter out 'final_layers' weights due to mismatch in output channels (e.g. 3 → 1)
ckpt_dict = ckpt.get('model_state_dict', ckpt)

# ✅ Strip "module." prefix AND skip final_layers
stripped_ckpt = {
    k.replace('module.', ''): v
    for k, v in ckpt_dict.items()
    if not k.replace('module.', '').startswith('final_layers')
}

model.load_state_dict(stripped_ckpt, strict=False)
model = model.to(DEVICE)
model = nn.DataParallel(model)

print(f"[INFO] # Trainable params: {count_params(model)}")

# Freeze everything except stage4 and final layers
for name, param in model.named_parameters():
    param.requires_grad = any(k in name for k in ['stage4', 'final_layers'])

# ----------------------------
# OPTIMIZER & LOSS
# ----------------------------
criterion = HeatmapLoss(CONFIG)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# ----------------------------
# TRAIN + VALIDATION LOOP
# ----------------------------
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    print(f"\n[Epoch {epoch}/{EPOCHS}]")

    # Train
    train_result = train_epoch(epoch, model, train_loader, criterion, optimizer, DEVICE)

    # Validate using test_epoch logic
    val_result = test_epoch(epoch, model, val_loader, criterion, DEVICE, cfg=CONFIG)

    val_loss = val_result['loss']
    print(f"(VAL) Epoch {epoch} Loss: {val_loss:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            {
                'model_state_dict': model.module.state_dict(),
                'epoch': epoch,
                'train_loss': train_result['loss'],
                'val_loss': val_loss
            },
            is_best=True,
            model_path=os.path.join(OUTPUT_DIR, f'checkpoint_ep{epoch}.pth.tar'),
            best_model_name='best_finetuned.pth.tar'
        )
        print(f"[Epoch {epoch}] ✅ Saved new best model (val_loss={val_loss:.6f})")

    scheduler.step()

