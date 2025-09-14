from __future__ import annotations
import json, math
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = "config.json"

# OpenPose BODY_25 indices
NOSE   = 0
REYE   = 15
LEYE   = 16
REAR   = 17
LEAR   = 18
RANKLE = 11   # BODY_25 right ankle
LANKLE = 14   # BODY_25 left ankle
RHEEL  = 24
LHEEL  = 21
RBIG   = 22
LBIG   = 19

CONF_THRESH    = 0.10
RIM_TO_FLOOR_M = 3.05  # official rim height

# ---------- helpers ----------
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

def _angle_from_vertical(rim: Tuple[float,float], shadow: Tuple[float,float]) -> Tuple[float,float,float]:
    rx, ry = rim; sx, sy = shadow
    dx, dy = sx - rx, sy - ry
    ang_deg = math.degrees(math.atan2(abs(dx), abs(dy))) if abs(dy) > 1e-6 else 90.0
    return ang_deg, dx, dy

def _pick_best_scale_row(rows: List[Dict[str,Any]], require_vertical_ok: bool) -> Optional[Dict[str,Any]]:
    """Choose the row with the smallest angle to vertical where shadow is below rim."""
    best = None
    best_ang = 1e9
    for r in rows or []:
        if require_vertical_ok and not r.get("vertical_ok", False):
            continue
        rim = r.get("rim_top_mid")
        sh  = r.get("chosen")
        if not (isinstance(rim, (list,tuple)) and len(rim)==2 and
                isinstance(sh,  (list,tuple)) and len(sh) ==2):
            continue
        ang, dx, dy = _angle_from_vertical((float(rim[0]), float(rim[1])),
                                           (float(sh[0]),  float(sh[1])))
        if dy <= 0:   # need shadow below rim
            continue
        if ang < best_ang:
            best_ang = ang
            best = {**r, "_angle_deg": ang, "_dx": dx, "_dy": dy}
    return best

def _extract_xyc(p: List[float], idx: int) -> Optional[Tuple[float,float,float]]:
    base = 3*idx
    if base+2 >= len(p): return None
    x, y, c = p[base:base+3]
    if x is None or y is None or c is None: return None
    x = float(x); y = float(y); c = float(c)
    if c < CONF_THRESH: return None
    return (x, y, c)

def _foot_y(person_p: List[float]) -> Optional[float]:
    """Use the lowest available foot/ankle keypoint (largest y in image)."""
    ys = []
    for idx in (RHEEL, LHEEL, RBIG, LBIG, RANKLE, LANKLE):
        v = _extract_xyc(person_p, idx)
        if v:
            ys.append(v[1])
    return max(ys) if ys else None

def _eye_or_nose_y(person_p: List[float]) -> Tuple[Optional[float], str]:
    """Prefer the higher (smaller y) of the two eyes; fallback to nose."""
    print(f"pose: {person_p}")
    eye_ys = []
    ear_ys = []
    for idx in (REYE, LEYE):
        v = _extract_xyc(person_p, idx)
        if v:
            eye_ys.append(v[1])
    if eye_ys:
        return (min(eye_ys), "eye")
    for idx in (REAR, LEAR):
        v = _extract_xyc(person_p, idx)
        if v:
            ear_ys.append(v[1])
    if ear_ys:
        return (min(ear_ys), "eye") #eye and ear approx same ratio
    v = _extract_xyc(person_p, NOSE)
    return (v[1], "nose") if v else (None, "none")

# ---------- main estimator ----------
class HeightEstimator:
    """
    Uses the straightest rim→shadow row in the event to set pixel→meter scale (3.05 m),
    then measures the scorer’s height on the shooter frame only.
    """
    def __init__(self,
                 config_path: str = DEFAULT_CONFIG_PATH,
                 require_vertical_ok_for_scale: bool = False,
                 use_eye_ratio: bool = True,
                 eye_to_height_ratio: float = 0.93,
                 nose_to_height_ratio: float = 0.96,
                 eye_to_vertex_add_m: float = 0.12,
                 nose_to_vertex_add_m: float = 0.10):
        with open(config_path, "r", encoding="utf-8") as f:
            C = json.load(f)
        self.pose_jsonl_path = C["pose"]["out_jsonl"]

        self.require_vertical_ok_for_scale = bool(require_vertical_ok_for_scale)
        self.use_eye_ratio = bool(use_eye_ratio)
        self.eye_to_height_ratio = float(eye_to_height_ratio)
        self.nose_to_height_ratio = float(nose_to_height_ratio)
        self.eye_to_vertex_add_m = float(eye_to_vertex_add_m)
        self.nose_to_vertex_add_m = float(nose_to_vertex_add_m)

    def estimate_for_event(self,
                           scoring_event: Dict[str,Any],
                           shadow_rows: List[Dict[str,Any]],
                           shooter_result: Dict[str,Any]) -> Optional[Dict[str,Any]]:
        """
        Returns:
          {
            event_id, best_scale_frame, angle_deg, dy_px, scale_px_per_m,
            shooter_frame, person_index, seg_type, seg_dy_px,
            est_height_m, est_height_cm
          }
        or None if insufficient data.
        """
        ev_id = scoring_event.get("event_id") or scoring_event.get("scoring_event_count")

        # 1) scale from the straightest rim→shadow line (any frame in event)
        best = _pick_best_scale_row(shadow_rows, self.require_vertical_ok_for_scale)
        if not best:
            print("return1")
            return None
        dy = float(best.get("_dy", 0.0))
        if dy <= 0:
            print("return2")
            return None
        px_per_m = dy / RIM_TO_FLOOR_M

        # 2) shooter frame + person
        shooter_frame = shooter_result.get("shooter_frame")
        pidx          = shooter_result.get("person_index")
        if shooter_frame is None or pidx is None:
            print("return3")
            return None

        window = _read_pose_window_for_event(ev_id, self.pose_jsonl_path)
        fr = next((f for f in (window.get("frames") or []) if f.get("frame") == shooter_frame), None)
        if not fr:
            print("return4")
            return None
        people = fr.get("people") or []
        pidx = int(pidx)
        if not (0 <= pidx < len(people)):
            print("return5")
            return None
        person = people[pidx]
        p = person.get("p")
        if not isinstance(p, list) or len(p) < 3*25:
            print("return6")
            return None

        # 3) ankle/foot → eye (or nose) vertical pixels, same frame
        y_foot = _foot_y(p)
        y_top, seg_type = _eye_or_nose_y(p)
        if y_foot is None or y_top is None:
            print(f"{y_foot} foot, {y_top} top")
            return None
        seg_dy_px = float(y_foot - y_top)
        if seg_dy_px <= 0:
            print("return8")
            return None

        # 4) convert to meters
        seg_m = seg_dy_px / px_per_m
        if self.use_eye_ratio:
            if seg_type == "eye":
                height_m = seg_m / max(1e-6, self.eye_to_height_ratio)
            else:  # nose
                height_m = seg_m / max(1e-6, self.nose_to_height_ratio)
        else:
            add_m = self.eye_to_vertex_add_m if seg_type == "eye" else self.nose_to_vertex_add_m
            height_m = seg_m + add_m

        return {
            "event_id": ev_id,
            "best_scale_frame": best.get("frame"),
            "angle_deg": round(float(best.get("_angle_deg", 0.0)), 2),
            "dy_px": round(float(best.get("_dy", dy)), 1),
            "scale_px_per_m": round(px_per_m, 4),
            "shooter_frame": shooter_frame,
            "person_index": pidx,
            "seg_type": seg_type,                 # "eye" or "nose"
            "seg_dy_px": round(seg_dy_px, 1),    # foot→eye/nose pixels
            "est_height_m": round(float(height_m), 3),
            "est_height_cm": int(round(float(height_m) * 100.0)),
        }

