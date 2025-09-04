# differentiate_points.py
# Minimal, import-only module (no CLI)

import os, glob, json, math
from typing import Dict, List, Tuple
from collections import defaultdict

# ---- indices / constants ----
# Court keypoint indices for free-throw lines (your spec)
FT_LEFT  = (8, 9)
FT_RIGHT = (16, 17)

# OpenPose BODY_25 ankle indices
R_ANKLE = 11
L_ANKLE = 14

# ---- tiny utils ----
def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate minor noise
                continue
    return out

def build_frame_order(frames_dir_rel: str, precomp_base: str = "../precompute") -> List[str]:
    """
    Returns list of frame file NAMES (e.g., 'frame_001.png'), sorted.
    Assumes frames at <precomp_base>/<frames_dir_rel>/*.png
    """
    frames_glob = os.path.join(precomp_base, frames_dir_rel, "*.png")
    frames = sorted(glob.glob(frames_glob))
    return [os.path.basename(p) for p in frames]

def point_line_distance(px: Tuple[float,float], a: Tuple[float,float], b: Tuple[float,float]) -> float:
    (x, y) = px; (x1, y1) = a; (x2, y2) = b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1)*dx + (y - y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    pxp, pyp = x1 + t*dx, y1 + t*dy
    return math.hypot(x - pxp, y - pyp)

def extract_ankle_xy(person_p_flat: List[float]) -> Tuple[Tuple[float,float], float]:
    """
    'p' is a flat list: [x0,y0,c0, x1,y1,c1, ...]
    Returns (best_ankle_xy, best_conf)
    """
    def get_xyc(idx: int):
        off = idx * 3
        if off + 2 >= len(person_p_flat):
            return (0.0, 0.0), 0.0
        return (float(person_p_flat[off]), float(person_p_flat[off+1])), float(person_p_flat[off+2])
    r_xy, r_c = get_xyc(R_ANKLE)
    l_xy, l_c = get_xyc(L_ANKLE)
    return (r_xy, r_c) if r_c >= l_c else (l_xy, l_c)

# ---- indexers (import-safe, no side effects) ----
def index_pose_by_frame(pose_jsonl_path: str) -> Dict[str, List[dict]]:
    """
    Returns: frame_name -> list of 'people' dicts (each has 'p')
    Accepts:
      A) {"frame":"...", "people":[{"p":[...]}, ...]}
      B) {"event_id":..., "frames":[{"frame":"...", "people":[...]}, ...]}
    """
    idx = defaultdict(list)
    for row in load_jsonl(pose_jsonl_path):
        if "frames" in row:
            for fr in row["frames"]:
                idx[fr["frame"]] = fr.get("people", [])
        elif "frame" in row:
            idx[row["frame"]] = row.get("people", [])
    return idx

def index_court_by_frame(court_jsonl_path: str) -> Dict[str, List[List[float]]]:
    """
    Returns: frame_name -> list of keypoints [ [x,y,c], ... ] (first set if nested)
    Assumes rows like: {"frame":"...", "keypoints":[[...]]}
    """
    idx = {}
    for row in load_jsonl(court_jsonl_path):
        fr = row.get("frame")
        kps = row.get("keypoints", [])
        if fr and kps:
            idx[fr] = kps[0]
    return idx

def _line_from_kps(kps: List[List[float]], i: int, j: int) -> Tuple[Tuple[float,float], Tuple[float,float], float]:
    if not kps or i >= len(kps) or j >= len(kps):
        return (0,0), (0,0), 0.0
    x1,y1,c1 = kps[i]
    x2,y2,c2 = kps[j]
    return (x1,y1), (x2,y2), min(float(c1), float(c2))

# ---- core decision ----
def decision_for_event(
    event: dict,
    frame_order: List[str],
    pose_idx: Dict[str, List[dict]],
    court_idx: Dict[str, List[List[float]]],
    ankle_dist_thresh_px: float = 50.0,
    min_ft_hits: int = 2,
    frame_gap_3pt: int = 50,
    debug: bool = True
) -> dict:
    """
    Rules:
      - 3PT: abs(idx(shooter) - idx(trigger)) > frame_gap_3pt
      - 1PT: in frames {shot-1, shot, shot+1}, ankle-to-FT-line distance <= threshold
             in at least min_ft_hits frames (either FT line)
      - 2PT: otherwise
    Requires keys (in event or merged event+result):
      trigger_frame, shooter_frame (or frame), person_index
    """
    trig = event.get("trigger_frame")
    shot = event.get("shooter_frame") or event.get("frame")
    person_ix = int(event.get("person_index", -1))

    if trig is None or shot is None or person_ix < 0:
        return {**event, "label": "UNKNOWN", "reason": "missing_fields"}

    if trig not in frame_order or shot not in frame_order:
        return {**event, "label": "UNKNOWN", "reason": "frames_not_in_order"}

    trig_idx = frame_order.index(trig)
    shot_idx = frame_order.index(shot)
    gap = abs(shot_idx - trig_idx)

    if debug:
        print(f"[Event {event.get('event_id','?')}] trigger={trig}({trig_idx}) shooter={shot}({shot_idx}) gap={gap}")

    # 3-pointer
    if gap > frame_gap_3pt:
        if debug:
            print(f"  -> 3PT (gap {gap} > {frame_gap_3pt})")
        return {**event, "label":"3PT", "reason": f"gap>{frame_gap_3pt}", "gap": gap}

    """
    # 1-pointer window: shot-1, shot, shot+1
    ft_hits = 0
    window = [i for i in (shot_idx-1, shot_idx, shot_idx+1) if 0 <= i < len(frame_order)]
    for idx in window:
        fr = frame_order[idx]
        people = pose_idx.get(fr, [])
        if not people or person_ix >= len(people):
            if debug: print(f"  [ft] {fr}: person {person_ix} missing")
            continue

        p = people[person_ix].get("p")
        if not p:
            if debug: print(f"  [ft] {fr}: no pose data")
            continue

        (ankle_xy, conf) = extract_ankle_xy(p)
        if conf <= 0:
            if debug: print(f"  [ft] {fr}: ankle conf low/zero")
            continue

        court_kps = court_idx.get(fr)
        if not court_kps:
            if debug: print(f"  [ft] {fr}: missing court kps")
            continue

        a1, a2, c1 = _line_from_kps(court_kps, *FT_LEFT)
        b1, b2, c2 = _line_from_kps(court_kps, *FT_RIGHT)
        d_left  = point_line_distance(ankle_xy, a1, a2) if c1 > 0 else float("inf")
        d_right = point_line_distance(ankle_xy, b1, b2) if c2 > 0 else float("inf")
        d = min(d_left, d_right)

        if debug:
            print(f"  [ft] {fr}: ankle={ankle_xy} d_left={d_left:.1f} d_right={d_right:.1f} d_min={d:.1f}")

        if d <= ankle_dist_thresh_px:
            ft_hits += 1

    if ft_hits >= min_ft_hits:
        if debug:
            print(f"  -> 1PT (free-throw proximity hits {ft_hits} >= {min_ft_hits})")
        return {**event, "label":"1PT", "reason": f"ft_hits>={min_ft_hits}", "ft_hits": ft_hits}

    if debug:
        print("  -> 2PT (default)")
    return {**event, "label":"2PT", "reason": "default"}
    """

    # 1-pointer window: shot-1, shot, shot+1
    ft_hits = 0
    idx = shot_idx
    fr = frame_order[idx]
    people = pose_idx.get(fr, [])
    if not people or person_ix >= len(people):
        if debug: print(f"  [ft] {fr}: person {person_ix} missing")

    p = people[person_ix].get("p")
    if not p:
        if debug: print(f"  [ft] {fr}: no pose data")

    (ankle_xy, conf) = extract_ankle_xy(p)
    if conf <= 0:
        if debug: print(f"  [ft] {fr}: ankle conf low/zero")

    court_kps = court_idx.get(fr)
    if not court_kps:
        if debug: print(f"  [ft] {fr}: missing court kps")

    a1, a2, c1 = _line_from_kps(court_kps, *FT_LEFT)
    b1, b2, c2 = _line_from_kps(court_kps, *FT_RIGHT)
    d_left  = point_line_distance(ankle_xy, a1, a2) if c1 > 0 else float("inf")
    d_right = point_line_distance(ankle_xy, b1, b2) if c2 > 0 else float("inf")
    d = min(d_left, d_right)

    if debug:
        print(f"  [ft] {fr}: ankle={ankle_xy} d_left={d_left:.1f} d_right={d_right:.1f} d_min={d:.1f}")

    if d <= ankle_dist_thresh_px:
        ft_hits += 1

    if ft_hits >= min_ft_hits:
        if debug:
            print(f"  -> 1PT (free-throw proximity hits {ft_hits} >= {min_ft_hits})")
        return {**event, "label":"1PT", "reason": f"ft_hits>={min_ft_hits}", "ft_hits": ft_hits}

    if debug:
        print("  -> 2PT (default)")
    return {**event, "label":"2PT", "reason": "default"}

# What this module exposes
__all__ = [
    "FT_LEFT", "FT_RIGHT",
    "R_ANKLE", "L_ANKLE",
    "load_jsonl",
    "build_frame_order",
    "index_pose_by_frame",
    "index_court_by_frame",
    "decision_for_event",
]

