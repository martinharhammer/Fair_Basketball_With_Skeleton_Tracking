from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple, Optional
import cv2

DEFAULT_CONFIG_PATH = "config.json"

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

def _event_rows_from_list(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map frame -> row for rows already in memory."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        fn = r.get("frame")
        if fn:
            out[fn] = r
    return out

def _event_rows_from_file(path: str, ev_key: str) -> Dict[str, Dict[str, Any]]:
    """Scan JSONL once and return only rows for this event_id (as strings)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            eid = obj.get("event_id") or obj.get("scoring_event_count")
            fr  = obj.get("frame")
            if fr and str(eid) == ev_key:
                out[fr] = obj  # last occurrence per frame wins
    return out

class DistanceToHoopDrawer:
    """
    Draws, for a given scoring event's window frames:
      - red dot at hoop shadow point (from hoop_shadow rows)
      - yellow line from precomputed rim_top_mid to shadow point
    """

    def __init__(self,
                 config_path: str = DEFAULT_CONFIG_PATH,
                 out_root: Optional[str] = None,
                 require_vertical_ok: bool = False,
                 circle_radius: int = 5,
                 line_thickness: int = 2):
        with open(config_path, "r", encoding="utf-8") as f:
            C = json.load(f)

        self.frames_dir = C["frames_dir"]
        self.pose_jsonl_path = C["pose"]["out_jsonl"]
        self.shadow_jsonl_path = (C.get("hoop_shadow", {}) or {}).get("out_jsonl") \
                                 or "../precompute/output/hoop_shadow_points.jsonl"

        self.out_root = out_root or "../precompute/output/hoop_shadow_viz"

        self.require_vertical_ok = bool(require_vertical_ok)
        self.circle_radius = int(circle_radius)
        self.line_thickness = int(line_thickness)

        # Per-event in-memory cache: { ev_key -> { frame -> row } }
        self._event_cache: Dict[str, Dict[str, Any]] = {}

    def _get_points_from_row(self, row: Dict[str, Any]) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
        """Return (shadow_pt, rim_top_mid) from one row, honoring require_vertical_ok."""
        if self.require_vertical_ok and not row.get("vertical_ok", False):
            return None, None

        # shadow: prefer 'chosen', fallback to LH/RH
        sp = row.get("chosen")
        if not (isinstance(sp, (list, tuple)) and len(sp) == 2):
            for k in ("LH", "RH"):
                pt = row.get(k)
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    sp = pt
                    break

        rp = row.get("rim_top_mid")
        if not (isinstance(rp, (list, tuple)) and len(rp) == 2):
            return None, None

        if not (isinstance(sp, (list, tuple)) and len(sp) == 2):
            return None, None

        return (float(sp[0]), float(sp[1])), (float(rp[0]), float(rp[1]))

    def draw_for_event(self,
                       scoring_event: Dict[str, Any],
                       shadow_rows: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        If shadow_rows is provided (fresh from compute_for_event), use it directly (fastest).
        Otherwise, lazily load/cached rows for this event from JSONL once.
        """
        ev_id  = scoring_event.get("event_id") or scoring_event.get("scoring_event_count")
        ev_key = str(ev_id)

        # Build per-event mapping once
        if shadow_rows is not None:
            self._event_cache[ev_key] = _event_rows_from_list(shadow_rows)
        elif ev_key not in self._event_cache:
            self._event_cache[ev_key] = _event_rows_from_file(self.shadow_jsonl_path, ev_key)

        rows_by_frame = self._event_cache.get(ev_key, {})
        if not rows_by_frame:
            return []

        # Get the window frames
        window = _read_pose_window_for_event(ev_id, self.pose_jsonl_path)
        frames = [fr.get("frame") for fr in window.get("frames", []) if fr.get("frame")]
        if not frames:
            return []

        out_dir = os.path.join(self.out_root, ev_key)
        os.makedirs(out_dir, exist_ok=True)

        written: List[str] = []
        for fname in frames:
            row = rows_by_frame.get(fname)
            if not row:
                continue

            shadow_pt, rim_pt = self._get_points_from_row(row)
            if shadow_pt is None or rim_pt is None:
                continue

            img_path = os.path.join(self.frames_dir, fname)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            sx, sy = int(round(shadow_pt[0])), int(round(shadow_pt[1]))
            rx, ry = int(round(rim_pt[0])),    int(round(rim_pt[1]))

            cv2.circle(img, (sx, sy), self.circle_radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.line(img, (rx, ry), (sx, sy), (0, 255, 255), thickness=self.line_thickness, lineType=cv2.LINE_AA)

            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, img)
            written.append(out_path)

        return written

