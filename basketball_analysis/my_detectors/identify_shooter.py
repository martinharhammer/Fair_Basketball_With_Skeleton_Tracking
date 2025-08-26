# identify_shooter.py
import os, json, math
from typing import List, Tuple, Dict

BACKTRACK_WINDOW = 40
SKIP_FRAMES = 20
DIST_THRESH_PX = 35.0
CONF_THRESH = 0.20
WRIST_IDXS = [(4, "R_WRIST"), (7, "L_WRIST")]  # BODY_25

def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def _save_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def _iter_wrists(person: dict):
    body = person.get("pose_keypoints_2d")
    if not isinstance(body, list) or len(body) < 3 * 25:
        return
    for idx, label in WRIST_IDXS:
        base = 3 * idx
        x, y, c = body[base:base+3]
        if x is None or y is None or c is None or c < CONF_THRESH:
            continue
        yield float(x), float(y), label, float(c)

def _min_wrist_to_ball(people, ball_xy: Tuple[float, float]):
    bx, by = ball_xy
    best = (float("inf"), -1, "", (float("nan"), float("nan")))
    for pi, person in enumerate(people):
        for (x, y, label, c) in _iter_wrists(person):
            d = math.hypot(x - bx, y - by)
            if d < best[0]:
                best = (d, pi, label, (x, y))
    return best

def run_identify_shooter( 
    scoring_events_jsonl: str,
    ball_jsonl: str,
    openpose_dir: str,
    frames_dir: str,
):
    # Load
    events = _load_jsonl(scoring_events_jsonl)
    ball_by_frame = {row["frame"]: row for row in _load_jsonl(ball_jsonl)}
    frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    updated = []
    for ev in events:
        event_frame = ev.get("frame")
        if event_frame not in frame_to_idx:
            updated.append(ev)
            continue

        scored_idx = frame_to_idx[event_frame]
        start_idx = max(0, scored_idx - SKIP_FRAMES)
        stop_idx  = max(0, scored_idx - BACKTRACK_WINDOW)

        shooter_info = None
        for j in range(start_idx, stop_idx - 1, -1):
            f = frames[j]
            ball_ann = ball_by_frame.get(f)
            if not ball_ann or "coordinates" not in ball_ann:
                continue
            coords = ball_ann["coordinates"]
            if not coords or len(coords) < 2:
                continue
            bx, by = float(coords[0]), float(coords[1])

            stem = os.path.splitext(os.path.basename(f))[0]
            op_path = os.path.join(openpose_dir, f"{stem}_keypoints.json")
            if not os.path.isfile(op_path):
                continue
            try:
                js = json.load(open(op_path, "r"))
            except Exception:
                continue
            people = js.get("people", [])

            d, person_idx, wrist_label, wrist_xy = _min_wrist_to_ball(people, (bx, by))

            if d < DIST_THRESH_PX:
                shooter_info = {
                    "shooter_frame": f,
                    "shooter_person_index": person_idx,
                    "min_dist_px": float(d),
                    "wrist_label": wrist_label,
                    "wrist_xy": wrist_xy,
                    "ball_xy": [bx, by],
                }
                break

        if shooter_info:
            ev.update(shooter_info)
        updated.append(ev)

    # Overwrite file in-place
    _save_jsonl(scoring_events_jsonl, updated)
    print(f"[Shooter] Updated {len(updated)} events in {scoring_events_jsonl}")

