import os, json, math
from typing import Optional, Dict, Any, List, Tuple, Set

SKIP_FRAMES = 20
DIST_THRESH_PX = 35.0
CONF_THRESH = 0.20
WRIST_R_IDX = 4
WRIST_L_IDX = 7

DEFAULT_CONFIG_PATH = "config.json"

def iter_wrists_p(person: Dict[str, Any]):
    body = person.get("p")
    for idx in (WRIST_R_IDX, WRIST_L_IDX):
        base = 3 * int(idx)
        x, y, c = body[base:base+3]
        if x is None or y is None or c is None:
            continue
        c = float(c)
        if c < CONF_THRESH:
            continue
        label = "R_WRIST" if idx == WRIST_R_IDX else "L_WRIST"
        yield float(x), float(y), label, c

def min_wrist_to_ball(people: List[Dict[str, Any]], bx: float, by: float):
    best = (float("inf"), -1, "", (float("nan"), float("nan")))
    for pi, person in enumerate(people):
        for (x, y, label, c) in iter_wrists_p(person):
            d = math.hypot(x - bx, y - by)
            if d < best[0]:
                best = (d, pi, label, (x, y))
    return best

def identify_shooter_for_event(event_window: Dict[str, Any], ball_by_frame: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    frames = event_window.get("frames", [])
    if not frames:
        return None
    n = len(frames)
    start_idx = max(0, n - 1 - SKIP_FRAMES)
    for i in range(start_idx, -1, -1):
        frame_rec = frames[i]
        fname = frame_rec.get("frame")
        ball = ball_by_frame.get(fname)
        if not ball:
            continue
        coords = ball.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        bx, by = float(coords[0]), float(coords[1])
        people = frame_rec.get("people", [])
        d, pidx, wrist_label, (wx, wy) = min_wrist_to_ball(people, bx, by)
        if d < DIST_THRESH_PX:
            return {
                "event_id": event_window.get("event_id"),
                "trigger_frame": frames[-1].get("frame"),
                "shooter_frame": fname,
                "person_index": pidx,
                "wrist": wrist_label,
                "min_dist_px": round(float(d), 2),
                "wrist_xy": [round(wx, 1), round(wy, 1)],
                "ball_xy": [round(bx, 1), round(by, 1)],
            }
    return -1

def _read_pose_window_for_event(event_id: Any, pose_jsonl_path: str) -> Dict[str, Any]:
    with open(pose_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            eid = obj.get("event_id") or obj.get("scoring_event_count")
            if eid == event_id and obj.get("frames"):
                return obj
    return {}

def _collect_ball_for_frames(frame_ids: Set[str], ball_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    wanted = set(frame_ids)
    found: Dict[str, Dict[str, Any]] = {}
    with open(ball_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not wanted:
                break
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            fn = obj.get("frame")
            if fn in wanted:
                found[fn] = obj
                wanted.remove(fn)
    return found

class IdentifyShooter:
    def __init__(self):
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            C = json.load(f)
        self.ball_jsonl_path = C["ball"]["out_jsonl"]
        self.pose_jsonl_path = C["pose"]["out_jsonl"]

    def identify_shooter(self, scoring_event: Dict[str, Any]) -> Dict[str, Any]:
        ev_id = scoring_event.get("event_id") or scoring_event.get("scoring_event_count")
        window = _read_pose_window_for_event(ev_id, self.pose_jsonl_path)
        frame_ids = {fr.get("frame") for fr in window.get("frames", []) if fr.get("frame")}
        ball_by_frame = _collect_ball_for_frames(frame_ids, self.ball_jsonl_path)
        shooter = identify_shooter_for_event(window, ball_by_frame)
        return shooter or {"event_id": ev_id, "trigger_frame": scoring_event.get("frame"), "shooter": None}

