from __future__ import annotations
import os, json
from ultralytics import YOLO
from precompute.helpers.frame_source import FrameSource
from precompute.helpers.progress import ProgressLogger

def _run_batch(model, batch_imgs, batch_names, jf):
    results = model(batch_imgs, imgsz=960, conf=0.10, iou=0.60, verbose=False)
    for nm, res in zip(batch_names, results):
        for box in res.boxes:
            x, y, w, h = box.xywh[0].tolist()
            rec = {"frame": nm, "bbox": [x,y,w,h], "conf": float(box.conf[0])}
            jf.write(json.dumps(rec)+"\n")

def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    video_path = C.get("video_path")
    hoop_cfg   = C["hoop"]
    model_path = hoop_cfg["model"]
    out_path   = hoop_cfg["out_jsonl"]
    batch_sz   = int(hoop_cfg.get("batch_size", 16))

    src = FrameSource(video_path=video_path)
    total_frames = src.count or None
    print(f"[Ball] Input frames: {src.count}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model = YOLO(model_path)

    logger = ProgressLogger(prefix="Hoop", total=total_frames, log_every=50)

    with open(out_path, "w", encoding="utf-8") as jf:
        batch_imgs, batch_names = [], []

        for _, name, frame in src:
            batch_imgs.append(frame)
            batch_names.append(name)
            logger.tick()

            if len(batch_imgs) == batch_sz:
                _run_batch(model, batch_imgs, batch_names, jf)
                batch_imgs.clear(); batch_names.clear()

        if batch_imgs:
            _run_batch(model, batch_imgs, batch_names, jf)

    logger.done(f"Output saved: {out_path}")

if __name__ == "__main__":
    main()

