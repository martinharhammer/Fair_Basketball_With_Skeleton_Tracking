# game_logic/hoop_bbox_for_triggerframes.py
import os, json, cv2, re

CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config_video.json")

ROOT = os.path.abspath(os.path.dirname(__file__))
PRECOMPUTE_BASE = os.path.abspath(os.path.join(ROOT, "..", "precompute"))

def _abs_precompute(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PRECOMPUTE_BASE, p))

def _abs_here(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(ROOT, p))

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def get_video_geom(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if w <= 0 or fps <= 0:
        raise RuntimeError("Failed to read width/fps from video.")
    return w, fps

def frame_name_to_index(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0

def seconds_to_hms(t: float) -> str:
    # "HH:MM:SS.mmm"
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def hms_to_seconds(hms: str) -> float:
    if not hms:
        return 0.0
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    scoring_jsonl   = _abs_precompute(C["scoring"]["out_jsonl"])
    hoop_jsonl_path = _abs_precompute(C["hoop"]["out_jsonl"])
    video_path      = _abs_here(C["video_path"])

    # team sides config
    TS = C.get("team_sides") or {}
    base_left  = TS.get("left_side", "LEFT_TEAM")
    base_right = TS.get("right_side", "RIGHT_TEAM")
    switch     = bool(TS.get("switch_sides", False))
    halftime_s = hms_to_seconds(TS.get("halftime", ""))

    # video geometry
    frame_w, fps = get_video_geom(video_path)
    mid_x = 0.5 * frame_w

    out_path = os.path.join(PRECOMPUTE_BASE, "output", "hoop_bbox_per_event.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    # Build: frame -> {cx, cy, w, h}
    hoop_by_frame = {}
    for row in load_jsonl(hoop_jsonl_path):
        fr = row["frame"]
        cx, cy, w, h = map(float, row["bbox"])  # bbox = [cx, cy, w, h]
        hoop_by_frame[fr] = {"cx": cx, "cy": cy, "w": w, "h": h}

    with open(out_path, "a", encoding="utf-8") as out_f:
        for ev in load_jsonl(scoring_jsonl):
            ev_id = ev.get("event_id") or ev.get("scoring_event_count")
            trigger_frame = ev.get("frame") or ev.get("trigger_frame")
            hoop = hoop_by_frame.get(trigger_frame)

            # event time from frame index
            t_seconds = 0.0
            if trigger_frame:
                idx = frame_name_to_index(trigger_frame)
                t_seconds = idx / fps
            t_hms = seconds_to_hms(t_seconds)

            # sides at this event time (apply halftime swap if enabled)
            left_team, right_team = base_left, base_right
            if switch and (t_seconds >= halftime_s):
                left_team, right_team = base_right, base_left

            hoop_side = None
            offset_px = None
            team_assigned = "UNKNOWN"
            hoop_xyxy = None

            if hoop is not None:
                offset_px = hoop["cx"] - mid_x
                hoop_side = "RIGHT" if offset_px >= 0 else "LEFT"
                # Assign: LEFT hoop → right_side team; RIGHT hoop → left_side team
                team_assigned = right_team if hoop_side == "LEFT" else left_team

                # xyxy convenience
                x1 = hoop["cx"] - hoop["w"] * 0.5
                y1 = hoop["cy"] - hoop["h"] * 0.5
                x2 = hoop["cx"] + hoop["w"] * 0.5
                y2 = hoop["cy"] + hoop["h"] * 0.5
                hoop_xyxy = [x1, y1, x2, y2]

            rec = {
                "event_id": ev_id,
                "trigger_frame": trigger_frame,
                "timestamp_hms": t_hms,     # <-- added
                "event_time_s": round(t_seconds, 3),
                "hoop_cxcywh": hoop,        # {"cx","cy","w","h"} or None
                "hoop_xyxy": hoop_xyxy,     # [x1,y1,x2,y2] or None
                "frame_width": frame_w,
                "mid_x": mid_x,
                "offset_px": offset_px,     # cx - mid_x
                "hoop_side": hoop_side,     # "LEFT"/"RIGHT"/None
                "team_left_at_time": left_team,
                "team_right_at_time": right_team,
                "switch_sides": switch,
                "halftime_hms": TS.get("halftime", ""),
                "team_assigned": team_assigned
            }
            out_f.write(json.dumps(rec) + "\n")

    print(f"[OK] wrote: {out_path} (mid_x={mid_x:.1f}, frame_w={frame_w}, fps={fps:.3f})")

if __name__ == "__main__":
    main()

