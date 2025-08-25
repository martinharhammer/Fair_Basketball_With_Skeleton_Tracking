import cv2

class HoopBoxDrawer:
    """
    Draws bounding boxes around detected hoops.
    Expects per-frame detections:
      hoop_detections[frame_idx] = [
        {"bbox": [x1, y1, x2, y2], "conf": float},
        ...
      ]
    """

    def __init__(self):
        # defaults baked in
        self.color = (0, 0, 255)   # red (BGR)
        self.thickness = 1
        self.show_conf = False

    def draw(self, frames, hoop_detections):
        out = []
        for img, dets in zip(frames, hoop_detections):
            canvas = img.copy()
            for d in dets:
                x1, y1, x2, y2 = map(int, d["bbox"])
                h, w = canvas.shape[:2]
                # clamp coords
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.color, self.thickness)
                if self.show_conf and "conf" in d:
                    cv2.putText(
                        canvas, f"{d['conf']:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1, cv2.LINE_AA
                    )
            out.append(canvas)
        return out

