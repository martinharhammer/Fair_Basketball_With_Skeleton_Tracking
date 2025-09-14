from ultralytics import YOLO
import os, sys, json
from precompute.helpers.frame_source import FrameSource
from precompute.helpers.progress import ProgressLogger

def extract_keypoints(result):
    kps = result.keypoints
    if kps is None or kps.xy is None:
        return []
    xy = kps.xy
    conf = getattr(kps, "conf", None)
    out = []
    num_inst = xy.shape[0]
    num_pts  = xy.shape[1]
    for i in range(num_inst):
        inst = []
        for j in range(num_pts):
            x = float(xy[i, j, 0]); y = float(xy[i, j, 1])
            c = float(conf[i, j]) if conf is not None else None
            inst.append([x, y, c])
        out.append(inst)
    return out

def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    video_path = C.get("video_path")
    court_cfg  = C["court"]
    model_path = court_cfg["model"]
    out_path   = court_cfg["out_jsonl"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model = YOLO(model_path)

    batch_size = int(court_cfg.get("batch_size", 20))
    src = FrameSource(video_path=video_path)
    total_frames = src.count or None
    print(f"[Court] Input frames: {src.count}")
    if src.count == 0:
        sys.exit("[court] No frames found (video unreadable or folder empty).")

    logger = ProgressLogger(prefix="Court", total=total_frames, log_every=50)

    with open(out_path, "w", encoding="utf-8") as jf:
        batch_imgs, batch_names = [], []
        for _, name, frame in src:
            batch_imgs.append(frame)
            batch_names.append(name)
            logger.tick()
            if len(batch_imgs) == batch_size:
                results = model(batch_imgs, imgsz=960, conf=0.10, iou=0.60, verbose=False)
                for nm, res in zip(batch_names, results):
                    rec = {"frame": nm, "keypoints": extract_keypoints(res)}
                    jf.write(json.dumps(rec) + "\n")
                batch_imgs.clear(); batch_names.clear()

        if batch_imgs:
            results = model(batch_imgs, imgsz=960, conf=0.10, iou=0.60, verbose=False)
            for nm, res in zip(batch_names, results):
                rec = {"frame": nm, "keypoints": extract_keypoints(res)}
                jf.write(json.dumps(rec) + "\n")

    logger.done(f"Output saved: {out_path}")

if __name__ == "__main__":
    main()

