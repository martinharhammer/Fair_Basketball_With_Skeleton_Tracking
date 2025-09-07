# game_logic/assign_team.py
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np, cv2, os, json
import sys

# allow importing precompute/helpers without packaging gymnastics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "precompute")))
from helpers.frame_source import read_frame_at
from helpers.frame_utils import frame_name_to_index  # our local helper

MID_HIP, RHIP, LHIP = 8, 9, 12  # BODY_25

@dataclass
class TeamPrototype:
    name: str
    hue_deg: float   # OpenCV HSV hue in [0,180]
    sat_mean: float  # in [0,1]
    a_med: float     # OpenCV Lab a* [0,255]
    b_med: float     # OpenCV Lab b* [0,255]

def _circ_mean(h: np.ndarray) -> float:
    ang = h.astype(np.float32) * (np.pi / 90.0)
    s, c = np.sin(ang).mean(), np.cos(ang).mean()
    return float((np.degrees(np.arctan2(s, c)) / 2.0) % 180.0)

def _circ_std(h: np.ndarray) -> float:
    ang = h.astype(np.float32) * (np.pi / 90.0)
    s, c = np.sin(ang).mean(), np.cos(ang).mean()
    R = float(np.hypot(s, c))
    std_rad = np.sqrt(max(-2.0 * np.log(max(R, 1e-6)), 0.0))
    return float(np.degrees(std_rad) / 2.0)

def _sample_patch(img_bgr: np.ndarray, x: float, y: float, r: int,
                  min_sat_accept: float, max_hue_std: float):
    """
    Return (hue_deg, sat_0_1, a_med, b_med) or None if unusable.
    """
    H, W = img_bgr.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < W and 0 <= cy < H):
        return None

    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if mask.sum() < 40:
        return None

    crop = img_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    Hc, Sc, _ = cv2.split(hsv)
    _, Ac, Bc = cv2.split(lab)

    Hv = Hc[mask]
    Sv = Sc[mask].astype(np.float32) / 255.0
    Av = Ac[mask].astype(np.float32)
    Bv = Bc[mask].astype(np.float32)

    sat_mean = float(Sv.mean())
    hue_std  = _circ_std(Hv)

    if sat_mean > (min_sat_accept + 0.03) and hue_std > max_hue_std:
        return None

    return _circ_mean(Hv), sat_mean, float(np.median(Av)), float(np.median(Bv))

def _hue_d(h1: float, h2: float) -> float:
    d = abs(h1 - h2)
    return float(min(d, 180.0 - d))

def _dist(f, p: TeamPrototype) -> float:
    dH = _hue_d(f[0], p.hue_deg)
    dS = abs(f[1] - p.sat_mean)
    dA = abs(f[2] - p.a_med)
    dB = abs(f[3] - p.b_med)
    return float(np.sqrt((0.6 * dH) ** 2 + (1.0 * dS) ** 2 + (1.2 * dA) ** 2 + (1.0 * dB) ** 2))

class HipColorAssigner:
    """
    Robust team assigner using color near the 3 hip keypoints.
    Can load frames from a folder OR read specific frames from a video.
    """
    def __init__(self,
                 teamA: TeamPrototype,
                 teamB: TeamPrototype,
                 conf_thresh: float = 0.40,
                 patch_radius: int = 12,
                 offset_px: int = 3,
                 min_sat_accept: float = 0.05,
                 max_hue_std: float = 18.0,
                 video_path: str | None = None):
        self.video_path = video_path
        self.A, self.B = teamA, teamB
        self.conf_thresh = conf_thresh
        self.patch_radius = patch_radius
        self.offset_px = offset_px
        self.min_sat_accept = min_sat_accept
        self.max_hue_std = max_hue_std

    def _load_frame_bgr(self, frame_name: str):
        if self.video_path:
            idx = frame_name_to_index(frame_name)
            return read_frame_at(self.video_path, idx)

    def assign_for_frame(self,
                         frame_name: str,
                         people: List[dict],
                         person_index: int,
                         debug: bool = False):
        log = []
        if not (0 <= person_index < len(people)):
            log.append(f"bad index: person_index={person_index}, len(people)={len(people)}")
            return ("UNKNOWN", 1.0, log) if debug else ("UNKNOWN", 1.0)

        img = self._load_frame_bgr(frame_name)
        if img is None:
            log.append(f"frame load failed: {frame_name}")
            return ("UNKNOWN", 1.0, log) if debug else ("UNKNOWN", 1.0)

        entry = people[person_index]
        p_flat = entry.get("p")
        if p_flat is None:
            log.append("people[person_index] missing 'p'")
            return ("UNKNOWN", 1.0, log) if debug else ("UNKNOWN", 1.0)

        pose = np.asarray(p_flat, np.float32).reshape(25, -1)
        if pose.shape[1] < 3:
            log.append(f"pose has {pose.shape[1]} channels, expected 3")
            return ("UNKNOWN", 1.0, log) if debug else ("UNKNOWN", 1.0)

        feats = []
        offsets = [(0,0), ( self.offset_px,0), (-self.offset_px,0),
                          (0, self.offset_px), (0,-self.offset_px)]
        for name, i in (("MID_HIP", MID_HIP), ("RHIP", RHIP), ("LHIP", LHIP)):
            x, y, c = pose[i]
            if c < self.conf_thresh:
                log.append(f"{name}: conf {c:.2f} < {self.conf_thresh}")
                continue

            accepted_one = False
            for dx, dy in offsets:
                stat = _sample_patch(img, x+dx, y+dy, self.patch_radius,
                                     self.min_sat_accept, self.max_hue_std)
                if stat is None:
                    continue
                h, s, a, b = stat
                log.append(f"{name}({dx:+d},{dy:+d}): hue={h:.1f} sat={s:.2f} a={a:.1f} b={b:.1f}")
                feats.append(stat)
                accepted_one = True
                break  # take first good offset for this joint

            if not accepted_one:
                log.append(f"{name}: no good patch in offsets")

        if not feats:
            log.append("no valid hip patches")
            return ("UNKNOWN", 1.0, log) if debug else ("UNKNOWN", 1.0)

        f = np.mean(np.asarray(feats, np.float32), axis=0)
        dA, dB = _dist(f, self.A), _dist(f, self.B)
        label = self.A.name if dA < dB else self.B.name
        conf_ratio = (max(dA, dB) / (min(dA, dB) + 1e-6))
        log.append(f"feat mean: hue={f[0]:.1f} sat={f[1]:.2f} a={f[2]:.1f} b={f[3]:.1f}")
        log.append(f"distA={dA:.3f}, distB={dB:.3f} -> {label}, conf_ratio={conf_ratio:.3f}")

        return (label, float(conf_ratio), log) if debug else (label, float(conf_ratio))

    def assign_from_people(self,
                           frame_name: str,
                           people: List[dict],
                           person_index: int,
                           debug: bool = False):
        return self.assign_for_frame(frame_name, people, person_index, debug=debug)

    def assign_from_pose_jsonl(self,
                               frame_name: str,
                               person_index: int,
                               pose_jsonl_path: str,
                               debug: bool = False):
        try:
            with open(pose_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("frame") == frame_name and "people" in row:
                        return self.assign_for_frame(frame_name, row.get("people") or [], person_index, debug=debug)
                    frs = row.get("frames")
                    if isinstance(frs, list):
                        for fr in frs:
                            if fr.get("frame") == frame_name and "people" in fr:
                                return self.assign_for_frame(frame_name, fr.get("people") or [], person_index, debug=debug)
        except Exception as e:
            if debug:
                return "UNKNOWN", 1.0, [f"pose_jsonl read error: {e}"]
        return ("UNKNOWN", 1.0, ["frame not found in pose_jsonl"]) if debug else ("UNKNOWN", 1.0)

