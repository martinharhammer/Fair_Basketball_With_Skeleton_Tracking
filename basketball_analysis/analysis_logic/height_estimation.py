# hoop_metrics.py
import math
from typing import List, Optional, Tuple, Dict, Any

# You can tweak these defaults later
MAX_ANGLE_FROM_VERTICAL_DEG = 10.0  # how "vertical" the rim->shadow line must be
MIN_DY_PX = 2.0                     # require some downward distance (shadow below rim)

def _rim_top_mid_from_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """b = (x1,y1,x2,y2) -> top-edge midpoint."""
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, y1)

def _verticality_ok(rim_xy: Tuple[float, float],
                    shadow_xy: Tuple[float, float],
                    max_angle_deg: float,
                    min_dy_px: float) -> bool:
    """
    Checks if the segment from rim to shadow is 'nearly vertical'.
    We treat vertical as dy >> dx. Angle-from-vertical = atan2(|dx|, |dy|).
    """
    rx, ry = rim_xy
    sx, sy = shadow_xy
    dx = abs(sx - rx)
    dy = abs(sy - ry)
    # Require the shadow to be below (or at least not significantly above) the rim
    if (sy - ry) < min_dy_px:
        return False
    # Angle from vertical (0 means perfectly vertical)
    if dy <= 1e-6:
        return False
    angle_deg = math.degrees(math.atan2(dx, dy))
    return angle_deg <= max_angle_deg

def measure_shadow_gap(
    hoop_bboxes_per_frame: List[Optional[Tuple[float, float, float, float]]],
    shadow_pts_per_frame:   List[Optional[Tuple[float, float]]],
    max_angle_from_vertical_deg: float = MAX_ANGLE_FROM_VERTICAL_DEG,
    min_dy_px: float = MIN_DY_PX,
) -> List[Optional[Dict[str, Any]]]:
    """
    For each frame i:
      - uses hoop bbox (xyxy) and shadow point (x,y)
      - computes distance from rim top-midpoint to shadow point
      - flags invalid if not nearly vertical

    Returns list of dicts (or None) aligned with frames:
      {
        "rim_xy": (rx, ry),
        "shadow_xy": (sx, sy),
        "dx": float,
        "dy": float,
        "dist_px": float,
        "angle_from_vertical_deg": float,
        "valid_vertical": bool
      }
    """
    assert len(hoop_bboxes_per_frame) == len(shadow_pts_per_frame), \
        "lists must have same length (one entry per frame)"

    out: List[Optional[Dict[str, Any]]] = []
    for bbox, shadow in zip(hoop_bboxes_per_frame, shadow_pts_per_frame):
        if bbox is None or shadow is None:
            out.append(None)
            continue

        rx, ry = _rim_top_mid_from_xyxy(bbox)
        sx, sy = shadow

        dx = float(sx - rx)
        dy = float(sy - ry)
        dist = math.hypot(dx, dy)

        # angle from vertical (0=vertical). If dy==0, treat as invalid verticality.
        angle_deg = 90.0
        if abs(dy) > 1e-6:
            angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))

        valid = _verticality_ok((rx, ry), (sx, sy),
                                max_angle_deg=max_angle_from_vertical_deg,
                                min_dy_px=min_dy_px)

        out.append({
            "rim_xy": (rx, ry),
            "shadow_xy": (sx, sy),
            "dx": dx,
            "dy": dy,
            "dist_px": dist,
            "angle_from_vertical_deg": angle_deg,
            "valid_vertical": valid,
        })
    return out

