# analysis_logic/hoop_shadow_point.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import os
import json
import math
import numpy as np

# Uses your existing homography wrapper
from tactical_view_converter.homography import Homography

# ---- Court geometry (must match TacticalViewConverter) ----
HOOP_OFFSET_M = 1.575   # rim center offset from baseline (m), NBA/FIBA ~1.575
COURT_LEN_M   = 28.0
COURT_WID_M   = 15.0  # not used directly here, but kept for clarity

# ---- Verticality thresholds (angle-only; no distance/length) ----
DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG = 10.0
DEFAULT_MIN_DY_PX = 2.0  # shadow must be below rim by at least this many pixels


# ------------------------------- Projector -------------------------------

class HoopShadowProjector:
    """
    Projects hoop 'ground-touch' points (LH/RH) from tactical space back to image
    space for ALL frames using inverse homography built from court keypoints.
    """

    def __init__(self, key_points: List[Tuple[float, float]], tactical_width: float, tactical_height: float):
        self.key_points = key_points
        self.tw = float(tactical_width)
        self.th = float(tactical_height)

        # Fixed tactical positions for the two hoops on the midline
        x_left  = (HOOP_OFFSET_M / COURT_LEN_M) * self.tw
        x_right = ((COURT_LEN_M - HOOP_OFFSET_M) / COURT_LEN_M) * self.tw
        y_mid   = self.th / 2.0

        self.left_hoop_tv  = np.array([[x_left,  y_mid]], dtype=np.float32)
        self.right_hoop_tv = np.array([[x_right, y_mid]], dtype=np.float32)

    def _project_one(self, frame_kps: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Project LH/RH tactical points to image for a single frame."""
        valid_idx = [i for i, (x, y) in enumerate(frame_kps) if x > 0 and y > 0]
        if len(valid_idx) < 4:
            return None, None

        src = np.array([frame_kps[i] for i in valid_idx], dtype=np.float32)           # image points
        dst = np.array([self.key_points[i] for i in valid_idx], dtype=np.float32)     # tactical points

        H = Homography(src, dst)  # image -> tactical
        lh_img = H.inverse_transform_points(self.left_hoop_tv)[0]
        rh_img = H.inverse_transform_points(self.right_hoop_tv)[0]
        return (float(lh_img[0]), float(lh_img[1])), (float(rh_img[0]), float(rh_img[1]))

    def project_all(self, court_keypoints_per_frame) -> List[Dict[str, Optional[Tuple[float, float]]]]:
        """
        Returns list per frame: {"LH": (x,y) or None, "RH": (x,y) or None}
        """
        out: List[Dict[str, Optional[Tuple[float, float]]]] = []
        for kp_obj in court_keypoints_per_frame:
            frame_shadow = {"LH": None, "RH": None}
            if kp_obj is None or getattr(kp_obj, "xy", None) is None or len(kp_obj.xy) == 0:
                out.append(frame_shadow); continue

            frame_kps = kp_obj.xy.tolist()[0]  # (1, N, 2) -> list[(x,y), ...]
            try:
                lh, rh = self._project_one(frame_kps)
                frame_shadow["LH"] = lh
                frame_shadow["RH"] = rh
            except Exception:
                pass  # keep Nones on failure
            out.append(frame_shadow)
        return out


# ---------------------------- Small helpers -----------------------------

def rim_top_mid_from_xyxy(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Top-edge midpoint of a bbox (x1,y1,x2,y2)."""
    x1, y1, x2, _ = bbox_xyxy
    return ((x1 + x2) * 0.5, y1)

def _xyxy_of(item) -> Optional[Tuple[float, float, float, float]]:
    """
    Normalize a per-frame hoop detection item into (x1,y1,x2,y2) or None.
    Accepts dict with keys "bbox_xyxy"/"xyxy"/"bbox", or a len-4 list/tuple, or None.
    """
    if not item:
        return None
    if isinstance(item, list):
        # if top-k list, take first
        if not item:
            return None
        item = item[0]
    if isinstance(item, dict):
        for k in ("bbox_xyxy", "xyxy", "bbox"):
            if k in item and isinstance(item[k], (list, tuple)) and len(item[k]) == 4:
                x1, y1, x2, y2 = item[k]
                return (float(x1), float(y1), float(x2), float(y2))
    if isinstance(item, (list, tuple)) and len(item) == 4:
        x1, y1, x2, y2 = item
        return (float(x1), float(y1), float(x2), float(y2))
    return None

def select_shadow_nearest_x(rim_mid: Tuple[float, float],
                            lh: Optional[Tuple[float, float]],
                            rh: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Choose LH or RH whose x is closest to rim_mid.x."""
    cands: List[Tuple[float, Tuple[float, float]]] = []
    if lh is not None: cands.append((abs(lh[0] - rim_mid[0]), lh))
    if rh is not None: cands.append((abs(rh[0] - rim_mid[0]), rh))
    return min(cands)[1] if cands else None

def verticality_ok(rim_mid: Tuple[float, float],
                   shadow_point: Tuple[float, float],
                   max_angle_from_vertical_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
                   min_dy_px: float = DEFAULT_MIN_DY_PX) -> Tuple[bool, float, float, float]:
    """
    Returns (ok, dx, dy, angle_from_vertical_deg).
    Angle from vertical = atan2(|dx|, |dy|). Requires shadow below rim by min_dy_px.
    """
    rx, ry = rim_mid
    sx, sy = shadow_point
    dx = sx - rx
    dy = sy - ry
    if dy < min_dy_px:
        return (False, dx, dy, 90.0)
    angle_deg = math.degrees(math.atan2(abs(dx), abs(dy))) if abs(dy) > 1e-6 else 90.0
    return (angle_deg <= max_angle_from_vertical_deg, dx, dy, angle_deg)


# -------------------------- JSONL entrypoint ---------------------------

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def compute_and_save_shadow_points(
    frame_files: List[str],
    hoop_detections: List[Any],
    court_keypoints_per_frame: List[Any],
    tactical_view_converter: Any,
    output_jsonl: str,
    angle_thresh_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
    min_dy_px: float = DEFAULT_MIN_DY_PX,
) -> None:
    """
    Compute LH/RH hoop shadow points for all frames, select the one nearest in X
    to the rim top-midpoint, check verticality, and write a JSONL artifact.

    Each line:
      {"frame": <str>, "LH": [x,y] or null, "RH": [x,y] or null,
       "chosen": [x,y] or null, "vertical_ok": <bool>}
    """

    # 1) normalize hoop bboxes
    hoop_bboxes_per_frame = [_xyxy_of(item) for item in hoop_detections]

    # 2) project LH/RH for all frames
    projector = HoopShadowProjector(
        key_points=tactical_view_converter.key_points,
        tactical_width=tactical_view_converter.width,
        tactical_height=tactical_view_converter.height,
    )
    shadows_lr = projector.project_all(court_keypoints_per_frame)  # [{"LH":..., "RH":...}, ...]

    # 3) build rows
    rows: List[Dict[str, Any]] = []
    for fname, bbox, sp in zip(frame_files, hoop_bboxes_per_frame, shadows_lr):
        if bbox is None or sp is None:
            rows.append({"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False})
            continue

        lh = sp.get("LH")
        rh = sp.get("RH")
        if lh is None and rh is None:
            rows.append({"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False})
            continue

        rim_mid = rim_top_mid_from_xyxy(bbox)
        chosen = select_shadow_nearest_x(rim_mid, lh, rh)
        ok = False
        if chosen is not None:
            ok, _, _, _ = verticality_ok(rim_mid, chosen, max_angle_from_vertical_deg=angle_thresh_deg, min_dy_px=min_dy_px)

        rows.append({
            "frame": fname,
            "LH": lh,
            "RH": rh,
            "chosen": chosen,
            "vertical_ok": bool(ok),
        })

    # 4) write jsonl
    _write_jsonl(output_jsonl, rows)
    print(f"[HoopShadowPoint] wrote {len(rows)} rows -> {output_jsonl}")


# optional re-exports if you `from analysis_logic.hoop_shadow_point import *`
__all__ = [
    "HoopShadowProjector",
    "rim_top_mid_from_xyxy",
    "select_shadow_nearest_x",
    "verticality_ok",
    "compute_and_save_shadow_points",
]

