from ultralytics import YOLO
import cv2, glob, os, sys, json

# load config
CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    C = json.load(f)

# paths from config (interpreted relative to working directory)
frames_dir   = C["frames_dir"]
hoop_cfg     = C["hoop"]
model_path   = hoop_cfg["model"]
out_path   = hoop_cfg["out_jsonl"]

# gather images
img_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
print(f"[Hoop] Found {len(img_paths)} images")
if not img_paths:
    sys.exit("No images found.")

# load model
model = YOLO(model_path)

# open JSONL file for writing
with open(out_path, "w") as jf:
    print(f"[Debug-Hoop]: Opened {out_path}")
    # inference + save annotated
    for img_path in img_paths:
        result = model(img_path, imgsz=960, conf=0.10, iou=0.60, verbose=False)[0]

        # save annotated image
        #annotated_bgr = result.plot()  # already BGR
        #save_path = os.path.join("output", os.path.basename(img_path))
        #ok = cv2.imwrite(save_path, annotated_bgr)
        #print(("Saved " if ok else "Failed to save ") + save_path)
        
        # write detections to JSONL
        for box in result.boxes:
            # xywh in pixels
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            record = {
                "frame": os.path.basename(img_path),
                "bbox": [x, y, w, h],
                "conf": conf
            }
            jf.write(json.dumps(record) + "\n")

print(f"[Hoop] Detections saved to {out_path}")
