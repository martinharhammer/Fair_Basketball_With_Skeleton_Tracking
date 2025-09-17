#!/usr/bin/env python3
from __future__ import annotations
import os, json, cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import supervision as sv

# -------------------- Config helpers --------------------
def load_config(path="config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def frame_name_from_index(idx: int) -> str:
    return f"frame_{idx:06d}.png"

# -------------------- JSONL loaders --------------------
def load_court_by_frame(jsonl_path: str):
    out = {}
    if not os.path.exists(jsonl_path): return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            fn = obj.get("frame")
            if fn and "keypoints" in obj:
                out[fn] = obj["keypoints"]  # [[(x,y,conf), ...]] or (N,3)/(N,2)
    return out

def load_pose_by_frame(jsonl_path: str):
    """
    Supports either per-event windows (objects with 'frames') or per-frame entries.
    Produces mapping: frame -> list of people dicts (with 'p' flat list of BODY_25 triples).
    """
    out = {}
    if not os.path.exists(jsonl_path): return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            if "frames" in obj and isinstance(obj["frames"], list):
                for fr in obj["frames"]:
                    fn = fr.get("frame")
                    if fn:
                        out[fn] = fr.get("people", [])
            else:
                fn = obj.get("frame")
                if fn:
                    out[fn] = obj.get("people", [])
    return out

def load_hoop_shadow_by_frame(jsonl_path: str):
    """
    Expects lines like:
      {"frame": "...", "hoop": {"xywh":[x,y,w,h]}, "shadow":[sx,sy]}
    """
    out = {}
    if not os.path.exists(jsonl_path): return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            fn = obj.get("frame")
            if not fn: continue
            hoop_xywh = None
            if isinstance(obj.get("hoop"), dict):
                hoop_xywh = obj["hoop"].get("xywh")
            if hoop_xywh is None:
                hoop_xywh = obj.get("hoop_xywh")  # optional fallback
            out[fn] = {"hoop_xywh": hoop_xywh, "shadow": obj.get("shadow")}
    return out

# -------------------- Drawing helpers --------------------
class CourtKeypointDrawer:
    """
    Supervision-accurate look:
      • red vertex dots (radius=8)
      • white index labels (text_scale=0.5, thickness=1)
    Accepts either [[(x,y,conf), ...]] or np.array(N,3)/(N,2).
    """
    def __init__(self, keypoint_hex="#ff2c2c"):
        color = sv.Color.from_hex(keypoint_hex)
        self.vertex_annotator = sv.VertexAnnotator(
            color=color,
            radius=8
        )
        self.label_annotator = sv.VertexLabelAnnotator(
            color=color,
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

    @staticmethod
    def _xy_from_kps(kps_1xn3) -> np.ndarray:
        # Matches your JSONL structure and the Supervision example
        if isinstance(kps_1xn3, list) and len(kps_1xn3) > 0:
            arr = np.array(kps_1xn3[0], dtype=float)  # (N,3) or (N,2)
        else:
            arr = np.asarray(kps_1xn3, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        return arr[:, :2].astype(np.float32)

    def draw(self, frame: np.ndarray, kps_1xn3) -> np.ndarray:
        out = frame.copy()
        xy = self._xy_from_kps(kps_1xn3)
        if xy.size == 0:
            return out

        # Supervision expects a batch shape: (N_instances, K, 2)
        xy_batched = xy[np.newaxis, ...]           # (1, K, 2)
        keypoints = sv.KeyPoints(xy=xy_batched)

        # Draw dots
        out = self.vertex_annotator.annotate(scene=out, key_points=keypoints)

        # Draw labels – for a single instance, pass a flat list of length K
        labels = [str(i) for i in range(xy.shape[0])]
        out = self.label_annotator.annotate(scene=out, key_points=keypoints, labels=labels)
        return out

# BODY_25 topology (OpenPose)
BODY25_PAIRS = [
    (0,1), (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (1,8), (8,9), (9,10), (10,11),
    (8,12), (12,13), (13,14),
    (0,15), (15,17), (0,16), (16,18),
    (14,19), (19,20), (14,21),
    (11,22), (22,23), (11,24)
]

# Approximate OpenPose color palette (BGR)
OPENPOSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

def color_for_pair(i: int) -> Tuple[int, int, int]:
    return OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]

class OpenPoseStylePoseDrawer:
    def __init__(self, conf_thresh=0.2, base_joint_radius=3, base_limb_thickness=2):
        self.th = conf_thresh
        self.jr = base_joint_radius
        self.lt = base_limb_thickness

    @staticmethod
    def _pts25_from_flat(p_flat: List[float]) -> np.ndarray:
        # (25,3): x,y,score
        return np.array(p_flat, dtype=float).reshape(-1, 3)[:25, :]

    def draw_people(self, frame: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        h, w = frame.shape[:2]
        # Scale thickness with resolution
        scale = max(1.0, min(h, w) / 720.0)
        jr = max(2, int(round(self.jr * scale)))
        lt = max(2, int(round(self.lt * scale)))

        out = frame.copy()
        for person in people or []:
            p = person.get("p")
            if not isinstance(p, list) or len(p) < 75:
                continue
            pts = self._pts25_from_flat(p)

            # limbs with per-connection color
            for i, (a, b) in enumerate(BODY25_PAIRS):
                xa, ya, ca = pts[a]; xb, yb, cb = pts[b]
                if ca >= self.th and cb >= self.th and xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)),
                             color_for_pair(i), lt, cv2.LINE_AA)

            # joints as colored circles (cycling palette)
            for j, (x, y, c) in enumerate(pts):
                if c >= self.th and x > 0 and y > 0:
                    cv2.circle(out, (int(x), int(y)), jr,
                               color_for_pair(j), -1, cv2.LINE_AA)
        return out

class HoopShadowDrawer:
    def __init__(self, hoop_color=(0,255,0), line_color=(0,255,255),
                 shadow_color=(0,0,255), hoop_thickness=2, line_thickness=2, shadow_radius=5):
        # Keep params for continuity; we won't draw the bbox.
        self.hoop_c = hoop_color
        self.line_c = line_color
        self.shadow_c = shadow_color
        self.hoop_t = hoop_thickness
        self.line_t = line_thickness
        self.shadow_r = shadow_radius

    @staticmethod
    def _rim_top_mid_from_xywh(xywh):
        x, y, w, h = map(float, xywh)
        return (int(round(x + w / 2.0)), int(round(y)))

    def draw(self, frame: np.ndarray, hoop_xywh, shadow_xy) -> np.ndarray:
        out = frame.copy()
        # ❌ No bbox drawing
        # ✅ Draw rim→shadow line + shadow dot when available
        if hoop_xywh and shadow_xy and len(shadow_xy) == 2:
            rx, ry = self._rim_top_mid_from_xywh(hoop_xywh)
            sx, sy = int(round(shadow_xy[0])), int(round(shadow_xy[1]))
            cv2.line(out, (rx, ry), (sx, sy), self.line_c, self.line_t, cv2.LINE_AA)
            cv2.circle(out, (sx, sy), self.shadow_r, self.shadow_c, -1, cv2.LINE_AA)
        return out

# -------------------- Main --------------------
def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    C = load_config(CONFIG_PATH)

    video_path   = C["video_path"]
    court_jsonl  = C["court"]["out_jsonl"]
    pose_jsonl   = C["pose"]["out_jsonl"]
    hoop_jsonl   = C["viz"]["height"]["frames_out_jsonl"]

    out_dir = C["viz"]["height"]["viz_path"]
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, "height_openpose_style.mp4")

    court_by_frame = load_court_by_frame(court_jsonl)
    pose_by_frame  = load_pose_by_frame(pose_jsonl)
    hoop_by_frame  = load_hoop_shadow_by_frame(hoop_jsonl)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    court_drawer = CourtKeypointDrawer(keypoint_hex="#ff2c2c")
    pose_drawer  = OpenPoseStylePoseDrawer(conf_thresh=0.2, base_joint_radius=3, base_limb_thickness=2)
    hoop_drawer  = HoopShadowDrawer(hoop_color=(0,255,0), line_color=(0,255,255),
                                    shadow_color=(0,0,255), hoop_thickness=2,
                                    line_thickness=2, shadow_radius=5)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fn = frame_name_from_index(idx)

        if fn in court_by_frame:
            frame = court_drawer.draw(frame, court_by_frame[fn])

        if fn in pose_by_frame:
            frame = pose_drawer.draw_people(frame, pose_by_frame[fn])

        if fn in hoop_by_frame:
            hs = hoop_by_frame[fn]
            frame = hoop_drawer.draw(frame, hs.get("hoop_xywh"), hs.get("shadow"))

        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    print(f"Saved video: {out_video}")

if __name__ == "__main__":
    main()

