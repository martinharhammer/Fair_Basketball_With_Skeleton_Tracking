from __future__ import annotations
import os, json, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..homography.homography import Homography
from ..utils.config import load_config

HOOP_OFFSET_M = 1.3
DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG = 10.0
DEFAULT_MIN_DY_PX = 2.0
DEFAULT_CONFIG_PATH = "config.json"

def rim_top_mid_from_cxcywh(b):
    xc, yc, _, h = map(float, b)
    return (xc, yc - 0.5 * h)

def _select_shadow_nearest_x(mx, lh, rh):
    c=[];
    if lh is not None: c.append((abs(lh[0]-mx), lh))
    if rh is not None: c.append((abs(rh[0]-mx), rh))
    return min(c)[1] if c else None

def _vertical_ok(rim_mid, sp, max_angle_deg, min_dy_px):
    rx,ry = rim_mid; sx,sy = sp
    dx,dy = sx-rx, sy-ry
    if dy < min_dy_px: return False
    ang = math.degrees(math.atan2(abs(dx), abs(dy))) if abs(dy)>1e-6 else 90.0
    return ang <= max_angle_deg

def _read_pose_window_for_event(event_id: Any, pose_jsonl_path: str) -> Dict[str, Any]:
    with open(pose_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            obj=json.loads(s)
            eid = obj.get("event_id") or obj.get("scoring_event_count")
            if obj.get("frames") and eid == event_id:
                return obj
    return {}

def _xywh_from_center(b):
    cx, cy, w, h = map(float, b)
    x = int(round(cx - w/2.0)); y = int(round(cy - h/2.0))
    return [x, y, int(round(w)), int(round(h))]

class HoopShadowForEvent:
    def __init__(self, tvc, config_path: str = DEFAULT_CONFIG_PATH,
                 angle_thresh_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
                 min_dy_px: float = DEFAULT_MIN_DY_PX):
        self.tvc = tvc
        self.angle_thresh_deg = angle_thresh_deg
        self.min_dy_px = min_dy_px

        C, resolve = load_config(config_path)
        self.pose_jsonl_path = resolve(C["pose"]["out_jsonl"])
        self.court_path      = resolve(C["court"]["out_jsonl"])
        self.hoop_path       = resolve(C["hoop"]["out_jsonl"])
        self.out_path        = resolve((C.get("hoop_shadow", {}) or {}).get("out_jsonl")
                                       or "precompute/output/hoop_shadow_points.jsonl")
        self.viz_out_path    = resolve(C["viz"]["height"]["frames_out_jsonl"])
        self.write_output    = bool(C["viz"]["height"]["visualize"])

        os.makedirs(os.path.dirname(self.viz_out_path) or ".", exist_ok=True)

        self._court_by_frame = self._index_by_frame(self.court_path)
        self._hoop_by_frame  = self._index_by_frame(self.hoop_path)

        self._tw = float(self.tvc.width)
        self._th = float(self.tvc.height)

        x_left  = (HOOP_OFFSET_M / float(self.tvc.actual_width_in_meters)) * self._tw
        x_right = ((float(self.tvc.actual_width_in_meters) - HOOP_OFFSET_M) / float(self.tvc.actual_width_in_meters)) * self._tw
        y_mid   = self._th / 2.0
        self._lh_tv = np.array([[x_left,  y_mid]], dtype=np.float32)
        self._rh_tv = np.array([[x_right, y_mid]], dtype=np.float32)

    @staticmethod
    def _index_by_frame(path: str) -> Dict[str, Dict[str, Any]]:
        out={}
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                s=line.strip()
                if not s: continue
                obj=json.loads(s); fn=obj.get("frame")
                if fn: out[fn]=obj
        return out

    def _project_one_frame(self, fname: str) -> Dict[str, Any]:
        court = self._court_by_frame.get(fname); hoop = self._hoop_by_frame.get(fname)
        if not court or not hoop:
            return {"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False, "hoop_xywh": None}

        kps = court.get("keypoints") or []
        if not kps or not kps[0]:
            return {"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False, "hoop_xywh": None}

        img_pts=[]; tac_pts=[]
        for i, kp in enumerate(kps[0]):
            x, y = float(kp[0]), float(kp[1])
            if x>0.0 and y>0.0:
                img_pts.append((x,y))
                tac_pts.append(tuple(map(float, self.tvc.key_points[i])))

        if len(img_pts) < 4:
            return {"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False, "hoop_xywh": None}

        try:
            H = Homography(np.array(img_pts, dtype=np.float32),
                           np.array(tac_pts, dtype=np.float32))
            lh_img = H.inverse_transform_points(self._lh_tv)[0]
            rh_img = H.inverse_transform_points(self._rh_tv)[0]
            lh = (float(lh_img[0]), float(lh_img[1]))
            rh = (float(rh_img[0]), float(rh_img[1]))
        except Exception:
            return {"frame": fname, "LH": None, "RH": None, "chosen": None, "vertical_ok": False, "hoop_xywh": None}

        bbox_c = hoop.get("bbox") if isinstance(hoop, dict) else None
        if not bbox_c or len(bbox_c) != 4:
            return {
                "frame": fname,
                "LH": [round(lh[0], 1), round(lh[1], 1)],
                "RH": [round(rh[0], 1), round(rh[1], 1)],
                "chosen": None,
                "vertical_ok": False,
                "hoop_xywh": None
            }

        rim_mid = rim_top_mid_from_cxcywh(bbox_c)
        chosen = _select_shadow_nearest_x(rim_mid[0], lh, rh)
        ok = bool(chosen and _vertical_ok(rim_mid, chosen, self.angle_thresh_deg, self.min_dy_px))
        xywh = _xywh_from_center(bbox_c)
        return {
            "frame": fname,
            "LH": [round(lh[0],1), round(lh[1],1)],
            "RH": [round(rh[0],1), round(rh[1],1)],
            "chosen": [round(chosen[0],1), round(chosen[1],1)] if chosen else None,
            "vertical_ok": ok,
            "rim_top_mid": [round(rim_mid[0], 1), round(rim_mid[1], 1)],
            "hoop_xywh": xywh
        }

    def compute_for_event(self, scoring_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        ev_id = scoring_event.get("event_id") or scoring_event.get("scoring_event_count")
        window = _read_pose_window_for_event(ev_id, self.pose_jsonl_path)
        frames = [fr.get("frame") for fr in window.get("frames", []) if fr.get("frame")]
        if not frames:
            return []

        rows = [self._project_one_frame(fn) for fn in frames]

        if self.write_output:
            with open(self.viz_out_path, "w", encoding="utf-8") as vout:
                for r in rows:
                    vout.write(json.dumps({
                        "frame": r.get("frame"),
                        "hoop": {"xywh": r.get("hoop_xywh")},
                        "shadow": r.get("chosen")
                    }) + "\n")

        return rows

