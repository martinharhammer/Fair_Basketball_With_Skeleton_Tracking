from ultralytics import YOLO
import glob, os, sys, json
from pathlib import Path

# config
CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    C = json.load(f)

frames_dir = C["frames_dir"]
court_cfg  = C["court"]
model_path = court_cfg["model"]          # <- add this in your config
out_path   = court_cfg["out_jsonl"]

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# gather frames (PNG)
img_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
print(f"[Court] Found {len(img_paths)} frames")
if not img_paths:
    sys.exit("[court] No images found.")

# load model
model = YOLO(model_path)

# helper: convert YOLO keypoints to pure-Python lists
def extract_keypoints(result):
    """
    Returns a list of instances; each instance is a list of [x, y, conf_or_None].
    If no keypoints: returns [].
    """
    kps = result.keypoints
    if kps is None or kps.xy is None:
        return []
    xy = kps.xy  # tensor: (num_instances, num_points, 2)
    conf = getattr(kps, "conf", None)  # (num_instances, num_points) or None

    out = []
    num_inst = xy.shape[0]
    num_pts  = xy.shape[1]
    for i in range(num_inst):
        inst = []
        for j in range(num_pts):
            x = float(xy[i, j, 0])
            y = float(xy[i, j, 1])
            c = float(conf[i, j]) if conf is not None else None
            inst.append([x, y, c])
        out.append(inst)
    return out

# inference (batched) + write JSONL
batch_size = int(court_cfg.get("batch_size", 20))
with open(out_path, "w", encoding="utf-8") as jf:
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        results = model(batch, imgsz=960, conf=0.10, iou=0.60, verbose=False)
        for frame_path, res in zip(batch, results):
            rec = {
                "frame": os.path.basename(frame_path),
                "keypoints": extract_keypoints(res)  # [] if none
            }
            jf.write(json.dumps(rec) + "\n")

print(f"[Court] Detections saved to {out_path}")

