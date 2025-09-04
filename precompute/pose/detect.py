import os, glob, json, sys
import cv2

# --- config ---
CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    C = json.load(f)

frames_dir   = C["frames_dir"]
pose_cfg     = C["pose"]
scoring_cfg  = C["scoring"]

openpose_root = pose_cfg["openpose_root"]
model_folder  = pose_cfg["model_folder"]
out_path      = pose_cfg["out_jsonl"]            # will contain one JSON object per scoring event
net_res       = pose_cfg["net_resolution"]       # e.g., "-1x368"
window_len    = int(pose_cfg.get("window_len", 90))  # frames to backtrack (inclusive)
scoring_path  = scoring_cfg["out_jsonl"]         # input: scoring detections JSONL
pose_vis = bool(pose_cfg.get("visualize", True))
viz_dir  = pose_cfg.get("viz_dir", "output/pose_viz")

if pose_vis:
    os.makedirs(viz_dir, exist_ok=True)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

# --- Bootstrap OpenPose Python package ---
sys.path.append(os.path.join(openpose_root, "build", "python"))
from openpose import pyopenpose as op  # will raise if not found

# --- Init OpenPose (BODY_25 only) ---
params = {
    "model_folder": model_folder,
    "model_pose": "BODY_25",
    "net_resolution": net_res,
    "hand": False,
    "face": False,
    "render_pose": 0,
    "display": 0,
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# --- Collect frames & index ---
frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
frames = [os.path.basename(p) for p in frame_paths]
idx_of = {f: i for i, f in enumerate(frames)}

print(f"[Pose] Found {len(frames)} frames")
if not frames:
    raise SystemExit(f"[Pose] No frames found in {frames_dir}")

# --- lower-body filter (BODY_25 indices) ---
# Right leg: hip=9, knee=10, ankle=11
# Left  leg: hip=12, knee=13, ankle=14
LOWER_IDX_LEFT  = (12, 13, 14)
LOWER_IDX_RIGHT = (9, 10, 11)
CONF_T = 0.20  # tweak as needed

def has_lower_body(person, conf_t=CONF_T, require_both_legs=False):
    """person: (25,3) array; return True if lower body is visible enough."""
    def vis(idxs):
        return sum(float(person[i, 2]) >= conf_t for i in idxs)
    left_vis  = vis(LOWER_IDX_LEFT)
    right_vis = vis(LOWER_IDX_RIGHT)
    return (left_vis >= 2 and right_vis >= 2) if require_both_legs else ((left_vis >= 2) or (right_vis >= 2))

# --- load scoring events (trigger frames) ---
events = []
with open(scoring_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        trig = rec["frame"]
        ev_id = rec.get("scoring_event_count") or rec.get("event_id")
        if ev_id is None:
            ev_id = len(events) + 1  # fallback if id not provided
        if trig in idx_of:
            events.append((ev_id, trig))

print(f"[Pose] Found {len(events)} scoring events")

# --- run windows & write one JSON object per event ---
with open(out_path, "w", encoding="utf-8") as jf:
    for ev_id, trig in events:
        t = idx_of[trig]
        start = max(0, t - (window_len - 1))  # inclusive
        end   = t                             # inclusive

        event_obj = {"event_id": ev_id, "frames": []}

        for i in range(start, end + 1):
            fname = frames[i]
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                event_obj["frames"].append({"frame": fname, "people": []})
                continue

            datum = op.Datum()
            datum.cvInputData = img
            v = op.VectorDatum()
            v.append(datum)
            opWrapper.emplaceAndPop(v)

            people_out = []
            if datum.poseKeypoints is not None:
                # (num_people, 25, 3)
                for person in datum.poseKeypoints:
                    if not has_lower_body(person, conf_t=CONF_T, require_both_legs=False):
                        continue
                    flat = []
                    for j in range(person.shape[0]):  # 25 keypoints
                        x = round(float(person[j, 0]), 1)
                        y = round(float(person[j, 1]), 1)
                        c = round(float(person[j, 2]), 2)
                        flat.extend([x, y, c])

                        if pose_vis and c is not None and c >= 0.2:
                            cv2.circle(img, (int(round(person[j, 0])), int(round(person[j, 1]))), 2, (0, 255, 0), -1)

                    people_out.append({"p": flat})

            event_obj["frames"].append({"frame": fname, "people": people_out})

            if pose_vis:
                save_name = f"ev{ev_id}_{fname}"
                save_path = os.path.join(viz_dir, save_name)
                ok = cv2.imwrite(save_path, img)
                if not ok:
                    print(f"[Pose][WARN] Failed to save {save_path}")

        jf.write(json.dumps(event_obj, separators=(",", ":")) + "\n")

print(f"[Pose] Windows saved to {out_path}")

