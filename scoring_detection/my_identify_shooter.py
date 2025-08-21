import os
import json
import math
import datetime as dt
from typing import Dict, List, Tuple, Optional

# ---------------- Config (EDIT THESE) ----------------
SCORING_EVENTS_JSONL = "output/scoring_event_detection/scoring_events.jsonl"
BALL_JSONL           = "001/ball/detections.jsonl"
OPENPOSE_DIR         = "001/players"
FRAMES_DIR           = "001/raw_frames"
OUTPUT_DIR           = "output/shooter_attribution"

BACKTRACK_WINDOW     = 40       # look back from the scoring frame (inclusive)
SKIP_FRAMES          = 20
DIST_THRESH_PX       = 35.0     # first time wrist-ball distance < threshold => shooter
CONF_THRESH          = 0.20     # ignore wrist keypoints below this confidence
# -----------------------------------------------------

WRIST_IDXS = [(4, "R_WRIST"), (7, "L_WRIST")]  # BODY_25

# ---------- helpers ----------
def load_scoring_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_jsonl_by_frame(path: str) -> Dict[str, dict]:
    d: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            key = ann.get("frame")
            if key is not None:
                d[key] = ann
    return d

def sort_frames(frames_dir: str) -> List[str]:
    imgs = [f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return sorted(imgs)

def stem_of_filename(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]

def read_openpose_for_frame(stem: str) -> Optional[dict]:
    """
    Expected naming exactly: frame_XXX_keypoints.json
    """
    path = os.path.join(OPENPOSE_DIR, f"{stem}_keypoints.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def iter_person_wrists(person: dict):
    """
    Yield (x, y, label, conf) for wrists only (BODY_25).
    Requires pose_keypoints_2d with >= 25 joints.
    """
    body = person.get("pose_keypoints_2d")
    if not isinstance(body, list) or len(body) < 3 * 25:
        return
    for idx, label in WRIST_IDXS:
        base = 3 * idx
        x, y, c = body[base:base+3]
        yield (x, y, label, c)

def min_wrist_to_ball_dist(people: List[dict], ball_xy: Tuple[float, float]) -> Tuple[float, int, str, Tuple[float, float]]:
    """
    Returns (min_dist, person_index, wrist_label, wrist_xy). If none, returns (inf, -1, "", (nan,nan)).
    """
    bx, by = ball_xy
    best = (float("inf"), -1, "", (float("nan"), float("nan")))
    for pi, person in enumerate(people):
        for (x, y, label, conf) in iter_person_wrists(person):
            if conf is not None and conf < CONF_THRESH:
                continue
            if x is None or y is None:
                continue
            d = math.hypot(x - bx, y - by)
            if d < best[0]:
                best = (d, pi, label, (x, y))
    return best

# ---------- main ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = os.path.join(OUTPUT_DIR, f"scoring_with_shooter_{stamp}.jsonl")
    summary = os.path.join(OUTPUT_DIR, f"summary_{stamp}.json")

    # inputs
    scoring_detections = load_scoring_rows(SCORING_EVENTS_JSONL)   # expects {"frame": "...", "scoring_event_count": N}
    ball_annos = load_jsonl_by_frame(BALL_JSONL)          # expects {"frame": "...", "coordinates": [x, y], ...}

    # frame order from disk (authoritative)
    frames = sort_frames(FRAMES_DIR)
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    events_with_shooter = 0

    with open(results, "w", encoding="utf-8") as fout:
        for scoring_detection in scoring_detections:
            
            event_frame = scoring_detection["frame"]                 # e.g., "frame_073.png"
            scoring_event_count = scoring_detection["scoring_event_count"] 

            if event_frame not in frame_to_idx:
                # scoring frame not found in frames_dir → skip gracefully
                continue

            scored_idx = frame_to_idx[event_frame]
            start_idx = max(0, scored_idx - SKIP_FRAMES)
            stop_idx = max(0, scored_idx - BACKTRACK_WINDOW)

            wrote = False
            best_overall = (float("inf"), -1, "", (float("nan"), float("nan")))
            best_overall_frame = None
            
            for j in range(start_idx, stop_idx - 1, -1):
                f = frames[j]
                     
                ball_ann = ball_annos.get(f)
                if not ball_ann:
                    continue
                coords = ball_ann.get("coordinates")
                if not coords or len(coords) < 2:
                    continue
                bx, by = float(coords[0]), float(coords[1])

                stem = stem_of_filename(f)
                js = read_openpose_for_frame(stem)  # expects "<stem>_keypoints.json"
                if not js or "people" not in js:
                    continue
                people = js.get("people", [])

                d, person_idx, wrist_label, wrist_xy = min_wrist_to_ball_dist(people, (bx, by))

                # as fallback
                if d < best_overall[0]:
                    best_overall = (d, person_idx, wrist_label, wrist_xy)
                    best_overall_frame = f

                if d < DIST_THRESH_PX:
                    out_line = {
                        "scoring_event_count": scoring_event_count,
                        "event_frame": event_frame,          # where score was detected
                        "shooter_frame": f,                   # where contact found
                        "shooter_person_index": person_idx,   # index in OpenPose "people"
                        "min_dist_px": float(d),
                        "below_threshold": True,
                        "threshold_px": DIST_THRESH_PX,
                        "wrist_label": wrist_label,           # "R_WRIST" / "L_WRIST"
                    }
                    fout.write(json.dumps(out_line) + "\n")
                    events_with_shooter += 1
                    wrote = True
                    break  # ← early exit as desired

            '''
            if not shooter_found:
                # fallback: closest in the window, even if above threshold
                min_d, person_idx, wrist_label, wrist_xy = best_overall
                out_line = {
                    "scoring_event_count": scoring_event_count,
                    "event_frame": event_frame,
                    "shooter_frame": best_overall_frame,
                    "shooter_person_index": person_idx,
                    "min_dist_px": None if math.isinf(min_d) else float(min_d),
                    "below_threshold": False,
                    "threshold_px": DIST_THRESH_PX,
                    "wrist_label": wrist_label,
                }
                fout.write(json.dumps(out_line) + "\n")
                if (best_overall_frame is not None) and (person_idx >= 0) and (not math.isinf(min_d)):
                    events_with_shooter += 1
          

    with open(summary, "w", encoding="utf-8") as fsum:
        json.dump(
            {
                "input": {
                    "scoring_events_jsonl": SCORING_EVENTS_JSONL,
                    "ball_jsonl": BALL_JSONL,
                    "openpose_dir": OPENPOSE_DIR,
                    "frames_dir": FRAMES_DIR,
                },
                "params": {
                    "backtrack_window": BACKTRACK_WINDOW,
                    "skip_frames": SKIP_FRAMES,
                    "distance_threshold_px": DIST_THRESH_PX,
                },
                "frames_count": len(frames),
                "scoring_events_in": total_scores,
                "events_with_shooter_output": events_with_shooter,
                "results_file": os.path.basename(results),
            },
            fsum,
            indent=2,
        )
    '''

    print(f"[INFO] wrote events -> {results}")
    # print(f"[INFO] wrote summary -> {summary}")

if __name__ == "__main__":
    main()
