# game_logic/assign_team_from_hoop.py
import os, json, re, cv2
from typing import Optional, Dict, Any

CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")

ROOT = os.path.abspath(os.path.dirname(__file__))
PRECOMPUTE_BASE = os.path.abspath(os.path.join(ROOT, "..", "precompute"))

def _abs_here(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(ROOT, p))

def _abs_precompute(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PRECOMPUTE_BASE, p))

def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def _frame_name_to_index(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0

def _hms_to_seconds(hms: str) -> float:
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

def _parse_bool(x: Any, default: bool=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return default

class HoopSideTeamAssigner:
    """
    Usage:
        assigner = HoopSideTeamAssigner()
        team = assigner.assign(trigger_frame="frame_012345.png")
        # or provide event time explicitly:
        team = assigner.assign("frame_012345.png", event_seconds=2134.8)
        team = assigner.assign("frame_012345.png", timestamp_hms="00:35:02")
    """
    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            C = json.load(f)

        self.video_path = _abs_here(C["video_path"])
        self.hoop_jsonl = _abs_precompute(C["hoop"]["out_jsonl"])

        # team sides config
        ts = C.get("team_sides") or {}
        self.base_left  = ts.get("left_side", "LEFT_TEAM")
        self.base_right = ts.get("right_side", "RIGHT_TEAM")
        self.switch     = _parse_bool(ts.get("switch_sides", False))
        self.halftime_s = _hms_to_seconds(ts.get("halftime", ""))

        # video geom
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps     = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if self.frame_w <= 0 or self.fps <= 0:
            raise RuntimeError("Failed to read width/fps from video.")
        self.mid_x = 0.5 * self.frame_w

        # build: frame -> {"cx","cy","w","h"}
        self.hoop_by_frame: Dict[str, Dict[str, float]] = {}
        for row in _load_jsonl(self.hoop_jsonl):
            fr = row["frame"]
            cx, cy, w, h = map(float, row["bbox"])  # bbox = [cx, cy, w, h]
            self.hoop_by_frame[fr] = {"cx": cx, "cy": cy, "w": w, "h": h}

    def assign(
        self,
        trigger_frame: str,
        *,
        event_seconds: Optional[float] = None,
        timestamp_hms: Optional[str] = None,
    ) -> str:
        """
        Returns team string based on hoop side at trigger_frame.
        Rule: hoop on LEFT  -> points to right_side team (at that time)
              hoop on RIGHT -> points to left_side team  (at that time)
        Applies halftime side switch if configured.

        event_seconds / timestamp_hms (optional):
          If provided, use these for halftime switching. Otherwise fallback to idx/fps.
        """
        if not trigger_frame:
            return "UNKNOWN"

        hoop = self.hoop_by_frame.get(trigger_frame)
        if hoop is None:
            return "UNKNOWN"

        # pick time source
        if event_seconds is not None:
            t_seconds = float(event_seconds)
        elif timestamp_hms:
            t_seconds = _hms_to_seconds(timestamp_hms)
        else:
            idx = _frame_name_to_index(trigger_frame)
            t_seconds = idx / self.fps if self.fps > 0 else 0.0

        # teams at this moment (apply halftime swap if needed)
        left_team, right_team = self.base_left, self.base_right
        if self.switch and (t_seconds >= self.halftime_s):
            left_team, right_team = right_team, left_team

        # side decision
        offset_px = hoop["cx"] - self.mid_x
        hoop_side = "RIGHT" if offset_px >= 0 else "LEFT"

        # map side -> team (note inversion per your rule)
        return right_team if hoop_side == "LEFT" else left_team

