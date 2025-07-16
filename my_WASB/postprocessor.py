import numpy as np
import cv2
import os
import glob

class WASBPostprocessor:
    """
    WASB-style postprocessor for Sports Ball Detection and Tracking:
    - Center-of-Heatmap (CoH) per blob
    - Simple online tracking (predict+filter)
    - Maintains a short history of ball positions
    - Adapts if detections are missed for several frames
    """

    def __init__(self, heatmap_threshold=0.2, distance_threshold=50, history_size=3, dist_weight=0.25, max_misses=5):
        self.heatmap_threshold = heatmap_threshold
        self.distance_threshold = distance_threshold
        self.history_size = history_size
        self.dist_weight = dist_weight
        self.max_misses = max_misses
        self.missed_frames = 0
        self.history = []
        self.mid = None

    def _extract_candidates(self, hm: np.ndarray):
        _, binary = cv2.threshold(hm, self.heatmap_threshold, 1, cv2.THRESH_BINARY)
        binary_uint8 = (binary * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        candidates = []
        for cnt in contours:
            mask = np.zeros_like(hm, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            values = hm[ys, xs]
            total = values.sum()
            cx = (xs * values).sum() / total
            cy = (ys * values).sum() / total
            confidence = total
            candidates.append((np.array([cx, cy]), confidence))
        return candidates

    def _predict_position(self):
        if len(self.history) >= 3:
            p2, p1, p0 = self.history[-3], self.history[-2], self.history[-1]
            a = p0 - 2*p1 + p2
            v = (p0 - p1) + a
            return p0 + v + a/2
        elif len(self.history) == 2:
            p1, p0 = self.history[-2], self.history[-1]
            return p0 + (p0 - p1)
        else:
            return None

    def run(self, preds, affine_mats):
        results = {0: {}}
        scale = list(preds.keys())[0]
        heatmaps = preds[scale].sigmoid_().cpu().numpy()
        _, N, _, _ = heatmaps.shape

        if self.mid is None:
            self.mid = N // 2

        pred_pos = self._predict_position()

        for j in range(N):
            hm = heatmaps[0, j]
            trans = affine_mats[0][0][j].cpu().numpy()

            if j == self.mid:
                cands = self._extract_candidates(hm)

                print(f'[DEBUG] Frame {j} — pred_pos = {pred_pos}, missed = {self.missed_frames}')
                for idx, (pos, conf) in enumerate(cands):
                    dist = np.linalg.norm(pos - pred_pos) if pred_pos is not None else -1
                    print(f'  → cand#{idx}: pos={pos}, conf={conf:.2f}, dist={dist:.2f}')

                if cands:
                    scored = []
                    for pos, conf in cands:
                        dist = np.linalg.norm(pos - pred_pos) if pred_pos is not None else 0
                        if pred_pos is not None and self.missed_frames < self.max_misses:
                            score = conf - self.dist_weight * dist
                        else:
                            score = conf
                        scored.append((pos, score, conf, dist))
                    pos, score, conf, dist = max(scored, key=lambda x: x[1])
                    xy = pos
                    self.history.append(xy)
                    if len(self.history) > self.history_size:
                        self.history.pop(0)
                    self.missed_frames = 0
                    xys, scores = [xy], [conf]  # use raw confidence for visualization
                else:
                    xys, scores = [], []
                    self.missed_frames += 1

                results[0][j] = {
                    scale: {'xys': xys, 'scores': scores, 'hm': hm, 'trans': trans}
                }
            else:
                results[0][j] = {
                    scale: {'xys': [], 'scores': [], 'hm': hm, 'trans': trans}
                }

        return results

    def save_video_from_overlays(self, overlay_dir, output_name='output_video.mp4', fps=30):
        overlay_files = sorted(glob.glob(os.path.join(overlay_dir, 'overlay_*.png')))
        if not overlay_files:
            print("[WARNING] No overlay images found to build video.")
            return

        first_frame = cv2.imread(overlay_files[0])
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(overlay_dir, output_name)
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for file in overlay_files:
            frame = cv2.imread(file)
            out.write(frame)

        out.release()
        print(f"[INFO] Video saved at: {video_path}")

