import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from datetime import datetime

from hrnet import HRNet
from preprocessing import ResizeWithEqualScale, SeqTransformCompose
from postprocessor import WASBPostprocessor   # assumes you copied their postprocessor.py

# --------------- CONFIGURATION ---------------
CONFIG_PATH   = 'config_hrnet.yaml'
#WEIGHTS_PATH  = 'finetune_outputs/best_full.pth.tar'
WEIGHTS_PATH = 'wasb_basketball_best.pth.tar'
FRAMES_FOLDER = 'videos/knicks_pacers/frames'
OUTPUT_FOLDER = 'outputs'       # base dir for everything

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

run_dir = os.path.join(
    OUTPUT_FOLDER,
    datetime.now().strftime('%d-%m-%y_%H-%M-%S')
)
os.makedirs(run_dir, exist_ok=True)

N_FRAMES = 3
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------- LOAD CONFIG ---------------
cfg = OmegaConf.load(CONFIG_PATH)

# --------------- BUILD MODEL ---------------
model = HRNet(cfg)
ckpt  = torch.load(WEIGHTS_PATH, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.to(DEVICE).eval()

# --------------- POSTPROCESSOR ---------------
#post = TracknetV2Postprocessor(score_threshold=cfg['detector']['postprocessor']['score_threshold'])
post = WASBPostprocessor(
    heatmap_threshold=0.2,       # same as your sigmoid threshold
    distance_threshold=80,       # tweak if needed
    history_size=3
)
# --------------- PREPROCESSOR ---------------
resize        = ResizeWithEqualScale(height=cfg['inp_height'],
                                     width= cfg['inp_width'])
seq_transform = SeqTransformCompose(frame_transform=resize)

# --------------- INFERENCE LOOP ---------------
frames = sorted([
    fn for fn in os.listdir(FRAMES_FOLDER)
    if fn.lower().endswith(('.jpg','.png'))
])

# build a “no‐op” affine matrix tensor so postprocessor will leave coordinates in image space
# shape: (batch=1, seq=N_FRAMES, 2, 3)
eye2x3 = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
affine_mats = {0: eye2x3.unsqueeze(0).unsqueeze(0).repeat(1, N_FRAMES, 1, 1)}

for i in tqdm(range(len(frames) - N_FRAMES + 1), desc='Inference'):
    # 1) load raw frames
    pil_frames = []
    for j in range(N_FRAMES):
        path = os.path.join(FRAMES_FOLDER, frames[i + j])
        img  = cv2.imread(path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(img))

    # 2) model input → heatmaps
    inp = seq_transform(pil_frames).unsqueeze(0).to(DEVICE)   # (1,3*N,H,W)
    with torch.no_grad():
        preds = model(inp)                                     # dict: scale→Tensor(1, N_FRAMES, H, W)

    # 3) postprocess → detections in original space
    results = post.run(preds, affine_mats)  # {batch_idx: {frame_idx: {scale: {...}}}
    scale    = list(preds.keys())[0]
    mid      = N_FRAMES // 2
    dets     = results[0][mid][scale]       # {'xys': [...], 'scores': [...], 'hm':..., 'trans':...}

    hm_max = np.max(dets['hm'])

    if len(dets['xys'])>0:
        xy, score = dets['xys'][0], dets['scores'][0]
    else:
        xy, score = None, None

    print(f'[Frame {i+mid}] hm_max: {hm_max:.4f}, xy: {xy}, score: {score}')

    # 4) raw heatmap for the middle frame
    hm = preds[scale][0, mid].sigmoid_().cpu().numpy()

    # 6) save the heatmap visualization into the run folder
    fig, ax = plt.subplots(figsize=(4,4))    # ← move this here
    ax.imshow(hm, cmap='hot')
    ax.axis('off')
    fig.tight_layout(pad=0)
    hm_fname = f'heatmap_{i+mid:06d}.png'
    fig.savefig(
        os.path.join(run_dir, hm_fname),
        dpi=150, bbox_inches='tight', pad_inches=0
    )
    plt.close(fig)

    # 7) overlay detection into the same run folder

    # load the original frame (in full resolution)
    orig = cv2.imread(os.path.join(FRAMES_FOLDER, frames[i+mid]))

    # map the (x,y) from network coords → original frame coords
    if xy is not None:
        orig_h, orig_w = orig.shape[:2]
        inp_h, inp_w   = cfg['inp_height'], cfg['inp_width']
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h
        xy = np.array([xy[0] * scale_x, xy[1] * scale_y])
        x_int, y_int = int(xy[0]), int(xy[1])

    # overlay the detection
    vis = orig.copy()
    if xy is not None:
        cv2.circle(vis, (x_int, y_int), 5, (0,0,255), thickness=-1)
        cv2.putText(
            vis, f'{score:.2f}', (x_int+6, y_int-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1,
            lineType=cv2.LINE_AA
        )
    vis_fname = f'overlay_{i+mid:06d}.png'
    cv2.imwrite(os.path.join(run_dir, vis_fname), vis)

    print(f'[{i+1}] saved → {hm_fname} + {vis_fname} in {run_dir}')

    # break if you only want the first
    # break

import cv2
import os
from glob import glob

# -------- CONFIG --------
overlay_folder = 'outputs/19-06-25_11-08-10'  # path to your overlay images
output_path = 'output_video.mp4'             # final video file name
fps = 30                                     # frames per second

# -------- LOAD & SORT IMAGES --------
image_files = sorted(glob(os.path.join(overlay_folder, 'overlay_*.png')))

# -------- SETUP VIDEO WRITER --------
# Read the first image to get frame size
frame = cv2.imread(image_files[0])
height, width, _ = frame.shape

# Use high-quality codec (H.264) with high bitrate for lossless effect
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'avc1' or 'libx264' if needed
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -------- WRITE FRAMES --------
for file in image_files:
    img = cv2.imread(file)
    video.write(img)

video.release()
print(f"[INFO] Video saved to: {output_path}")
