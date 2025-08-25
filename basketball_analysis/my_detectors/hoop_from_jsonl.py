import os, json, glob
import sys
sys.path.append('../')
from utils import read_stub, save_stub

def _xywh_center_to_xyxy(b):
    cx, cy, w, h = b
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [float(x1), float(y1), float(x2), float(y2)]

class HoopFromJSONL:
    """
    Loads hoop detections from a JSONL where bbox = [cx, cy, w, h] (pixels).
    Produces per-frame lists of {"bbox":[x1,y1,x2,y2], "conf": float}.
    """

    def __init__(self):
        pass  # keep constructor simple like your other classes

    def _read_jsonl_xywh_center(self, jsonl_path):
        per_image = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # tolerate prefixed line numbers
                p = line.find("{")
                if p > 0:
                    line = line[p:]
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fname = os.path.basename(rec.get("frame", "") or "")
                bbox = rec.get("bbox")
                conf = float(rec.get("conf", 1.0))
                if not fname or not bbox or len(bbox) != 4:
                    continue

                xyxy = _xywh_center_to_xyxy(bbox)
                per_image.setdefault(fname, []).append({"bbox": xyxy, "conf": conf})
        return per_image

    def get_detections(self, frames, read_from_stub=False, stub_path=None,
                       jsonl_path=None, raw_frames_dir=None, keep_top_k=1):
        if jsonl_path is None:
            raise ValueError("jsonl_path is required.")

        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        per_image = self._read_jsonl_xywh_center(jsonl_path)

        # try to align by names in `frames`
        names = []
        have_all_names = True
        for f in frames:
            if isinstance(f, str):
                names.append(os.path.basename(f))
            elif hasattr(f, "filename"):
                names.append(os.path.basename(getattr(f, "filename")))
            elif hasattr(f, "name"):
                names.append(os.path.basename(getattr(f, "name")))
            else:
                names.append(None); have_all_names = False

        results = []
        if have_all_names and per_image:
            for nm in names:
                dets = per_image.get(nm, [])
                if keep_top_k:
                    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)[:keep_top_k]
                results.append(dets)
        else:
            # fallback to ordering by raw frame filenames
            if raw_frames_dir is None:
                # if we can’t align, just produce empties of correct length
                results = [[] for _ in frames]
            else:
                pats = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
                img_paths = sorted(p for pat in pats for p in glob.glob(os.path.join(raw_frames_dir, pat)))
                basenames = [os.path.basename(p) for p in img_paths]
                for i in range(len(frames)):
                    nm = basenames[i] if i < len(basenames) else ""
                    dets = per_image.get(nm, [])
                    if keep_top_k:
                        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)[:keep_top_k]
                    results.append(dets)

        save_stub(stub_path, results)
        return results

