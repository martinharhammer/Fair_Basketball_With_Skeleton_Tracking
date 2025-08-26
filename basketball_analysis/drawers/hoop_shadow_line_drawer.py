# drawers/hoop_shadow_line_drawer.py
import os
import json
import cv2

class HoopShadowLineDrawer:
    """
    Minimal overlay: for each frame, if vertical_ok, draw a line from rim_top_mid
    to the shadow point (from hoop_shadow_points.jsonl). If rim_top_mid isn't in
    the JSONL, we compute it from the provided hoop_detections (xyxy).
    """

    def __init__(self, shadow_jsonl_path: str, color=(0, 255, 255), thickness=2, endpoints=False):
        self.color = color
        self.thickness = thickness
        self.endpoints = endpoints
        self.by_frame = self._load_jsonl(shadow_jsonl_path)

    @staticmethod
    def _load_jsonl(path: str):
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = os.path.basename(row.get("frame", ""))
                if key:
                    d[key] = row
        print(f"[HoopShadowLineDrawer] loaded {len(d)} rows from {path}")
        return d

    @staticmethod
    def _rim_top_mid_from_det(det):
        """
        det can be:
          - [x1,y1,x2,y2]
          - {"bbox":[x1,y1,x2,y2]} or {"xyxy":[x1,y1,x2,y2]}
        Returns (x, y) or None.
        """
        if det is None:
            return None
        if isinstance(det, (list, tuple)) and len(det) == 4:
            x1, y1, x2, y2 = det
        elif isinstance(det, dict):
            v = det.get("bbox") or det.get("xyxy")
            if not v or len(v) != 4:
                return None
            x1, y1, x2, y2 = v
        else:
            return None
        return ((float(x1) + float(x2)) * 0.5, float(y1))

    @staticmethod
    def _to_int_pt_in_bounds(p, img):
        x = int(round(p[0])); y = int(round(p[1]))
        h, w = img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            # clamp just in case
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
        return (x, y)

    def draw_shadow_circle(self, img, point, color=(0, 0, 255), radius=5):
        """
        Draw a filled circle at the given (float) point, clamped to image bounds.
        Returns the modified image.
        """
        x, y = self._to_int_pt_in_bounds(point, img)
        cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        return img

    def put_label(self, img, text, point, color=(0, 0, 255), offset=(6, -6), scale=0.5, thickness=1):
        """
        Put a small text label near the given point, using an (dx, dy) pixel offset.
        Returns the modified image.
        """
        x, y = self._to_int_pt_in_bounds(point, img)
        cv2.putText(
            img,
            str(text),
            (x + offset[0], y + offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return img

    def draw(self, frames, frame_files, hoop_detections=None):
        """
        Args:
            frames: list of numpy images (same order as frame_files)
            frame_files: list of frame filenames (basenames or paths)
            hoop_detections: optional list aligned with frame_files; each item either
                             None, a single xyxy box, a dict with 'xyxy'/'bbox',
                             or a list where the first element is the hoop box.

        Returns:
            list of images with lines drawn
        """
        out = []
        drawn = skipped = missing = 0

        for idx, (img, fname) in enumerate(zip(frames, frame_files)):
            canvas = img.copy()
            key = os.path.basename(fname)
            row = self.by_frame.get(key)

            if not row:
                missing += 1
                out.append(canvas)
                continue

            vertical_ok = bool(row.get("vertical_ok"))
            # your JSONL uses "chosen" for the selected shadow point
            shadow = row.get("shadow_point") or row.get("chosen")

            # get rim point: prefer JSONL, else compute from hoop_detections
            rim = row.get("rim_top_mid")
            if rim is None and hoop_detections is not None and idx < len(hoop_detections):
                det = hoop_detections[idx]
                # if it's a list of boxes, take the first
                if isinstance(det, list) and det and isinstance(det[0], (list, tuple, dict)):
                    det = det[0]
                rim = self._rim_top_mid_from_det(det)

            if vertical_ok and shadow and rim:
                p_shadow = self._to_int_pt_in_bounds(shadow, canvas)
                p_rim    = self._to_int_pt_in_bounds(rim, canvas)
                cv2.line(canvas, p_rim, p_shadow, self.color, self.thickness)
                if self.endpoints:
                    cv2.circle(canvas, p_rim, 3, self.color, -1)
                    cv2.circle(canvas, p_shadow, 3, self.color, -1)
                drawn += 1
            else:
                skipped += 1

            out.append(canvas)

        print(f"[HoopShadowLineDrawer] drew={drawn}, skipped={skipped}, no_match={missing}, total={len(out)}")
        return out

    def draw_with_markers(
        self,
        frames,
        frame_files,
        hoop_detections=None,
        circle_color=(0, 0, 255),
        label_color=(0, 0, 255),
        label_offset=(6, -6),
        match_eps=3.0,
    ):
        """
        Draw the vertical-ok line (via self.draw) and then overlay a red circle
        + label ("LH"/"RH"/"SH") at the chosen shadow point.

        Args:
            frames: list of images
            frame_files: list of frame basenames/paths (same order as frames)
            hoop_detections: optional, forwarded to self.draw(...)
            circle_color: BGR for the shadow point circle
            label_color: BGR for the label text
            label_offset: (dx, dy) pixels for label placement
            match_eps: distance threshold to decide if shadow == LH/RH
        """
        # First: draw the shadow→rim line
        canvases = self.draw(frames, frame_files, hoop_detections=hoop_detections)

        out = []
        for img, fname in zip(canvases, frame_files):
            row = self.by_frame.get(os.path.basename(fname))
            if not row or not row.get("vertical_ok"):
                out.append(img)
                continue

            shadow = row.get("shadow_point") or row.get("chosen")
            if not shadow:
                out.append(img)
                continue

            # Decide label based on which side it matches
            lab = "SH"
            lh = row.get("LH")
            rh = row.get("RH")

            def _close(a, b, eps=match_eps):
                return (
                    a is not None and b is not None and
                    ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5 <= eps
                )

            if _close(shadow, lh):
                lab = "LH"
            elif _close(shadow, rh):
                lab = "RH"

            img = self.draw_shadow_circle(img, shadow, color=circle_color, radius=5)
            img = self.put_label(img, lab, shadow, color=label_color, offset=label_offset)
            out.append(img)

        return out
