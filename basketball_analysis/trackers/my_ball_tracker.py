# ball_tracker.py

import os
import json
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import read_stub, save_stub


class BallTracker:
    """
    Loads ball annotations from a JSONL file (one JSON object per line with:
      - "frame": filename
      - "coordinates": [x, y] or null
      - "score": float or null (ignored)
    ) and produces tracks in the same shape as before:
        List[ Dict[int, {"bbox": List[float]}] ]
    where "bbox" now stores just a point [x, y].
    """

    def __init__(self, model_path=None):
        """
        `model_path` kept for compatibility; unused.
        """
        pass

    # ---------- JSONL loader ----------
    def load_annotations(self, anno_path):
        """
        Read JSONL annotations. Robust to leading line numbers (e.g., '  4 {...}').

        Returns:
            rows: list of records in file order
            by_name: dict mapping basename(frame) -> record
        """
        rows = []
        by_name = {}
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                brace_pos = line.find('{')
                if brace_pos > 0:
                    line = line[brace_pos:]
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
                if not isinstance(rec, dict):
                    continue
                frame = rec.get("frame")
                if not frame:
                    continue
                rows.append(rec)
                by_name[os.path.basename(frame)] = rec
        return rows, by_name

    def point_to_bbox(self, x, y):
        """
        For compatibility with downstream code expecting 'bbox',
        we store a 2D point as [x, y].
        """
        return [float(x), float(y)]

    # ---------- Build tracks from JSONL instead of running detection ----------
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, anno_path=None):
        """
        Build per-frame tracks using the JSONL annotations.

        Args:
            frames (list): list of frames (paths or arrays). We'll try to match by filename.
            read_from_stub (bool): whether to read cached results if available.
            stub_path (str): path to cache file.
            anno_path (str): path to JSONL annotations (required).

        Returns:
            list: per-frame dicts, each either {} or {1: {"bbox": [x, y]}}
        """
        if anno_path is None:
            raise ValueError("anno_path is required when loading annotations from file.")

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        rows, by_name = self.load_annotations(anno_path)

        # Extract basenames for matching if possible
        names = []
        all_strings = True
        for f in frames:
            name = None
            if isinstance(f, str):
                name = os.path.basename(f)
            elif hasattr(f, "filename"):
                name = os.path.basename(getattr(f, "filename"))
            elif hasattr(f, "name"):
                name = os.path.basename(getattr(f, "name"))
            else:
                all_strings = False
            names.append(name)

        use_by_name = all_strings and all(n is not None for n in names) and len(by_name) > 0

        tracks = []
        for i in range(len(frames)):
            tracks.append({})
            rec = None
            if use_by_name:
                rec = by_name.get(names[i])
            else:
                if i < len(rows):
                    rec = rows[i]

            if not rec:
                continue

            coords = rec.get("coordinates", None)
            if coords is None:
                continue

            x, y = float(coords[0]), float(coords[1])
            pt = self.point_to_bbox(x, y)  # returns [x, y]
            tracks[i][1] = {"bbox": pt}

        save_stub(stub_path, tracks)
        return tracks

    # ---------- Unchanged: distance gating ----------
    def remove_wrong_detections(self, ball_positions):
        """
        Filter out incorrect detections based on maximum allowed movement distance.
        Treats 'bbox' as [x, y] or [x1, y1, x2, y2]; only the first two values are used.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])
            if len(current_box) < 2:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    # ---------- Updated: interpolation supports 2 or 4 values ----------
    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing positions. Supports either 2-value points [x, y]
        or 4-value boxes [x1, y1, x2, y2]. Returns the same dimensionality it found.
        If an entire sequence has no valid numbers, the corresponding frames remain empty.
        """
        seq = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Detect dimensionality from first valid entry; default to 2
        dims = next((len(item) for item in seq if len(item) in (2, 4)), 2)

        # Normalize to fixed length with NaNs for missing
        norm = []
        for item in seq:
            if len(item) == dims:
                norm.append(item)
            else:
                norm.append([np.nan] * dims)

        cols = ['x', 'y'] if dims == 2 else ['x1', 'y1', 'x2', 'y2']
        df = pd.DataFrame(norm, columns=cols)

        # Interpolate then back/forward fill to close interior and edge gaps
        df = df.interpolate()
        df = df.bfill()
        df = df.ffill()

        out = []
        for i, row in df.iterrows():
            if row.isna().any():
                # No valid data anywhere for this row/column set; keep empty
                out.append({})
            else:
                out.append({1: {"bbox": [float(v) for v in row.tolist()]}})
        return out

