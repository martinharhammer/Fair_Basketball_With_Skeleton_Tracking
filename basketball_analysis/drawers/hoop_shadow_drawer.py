# hoop_shadow_drawer.py
import numpy as np
import cv2
from tactical_view_converter.homography import Homography

class HoopShadowDrawer:
    """
    Projects hoop 'ground touch' points from tactical view back into the camera
    and draws them as small rectangles with labels, using inverse homography.
    """
    def __init__(self, key_points, tactical_width, tactical_height,
                 color=(0, 0, 255), box=6, label_scale=0.5, label_thickness=1):
        self.key_points = key_points
        self.tw = tactical_width
        self.th = tactical_height
        self.color = color
        self.box = box
        self.label_scale = label_scale
        self.label_thickness = label_thickness

        # Rim center offset from baseline in meters (NBA/FIBA ~1.575 m).
        HOOP_OFFSET_M = 1.575
        COURT_LEN_M = 28.0   # matches TacticalViewConverter.actual_width_in_meters
        COURT_WID_M = 15.0   # matches TacticalViewConverter.actual_height_in_meters

        # Map to tactical coordinates
        x_left  = (HOOP_OFFSET_M / COURT_LEN_M) * self.tw
        x_right = ((COURT_LEN_M - HOOP_OFFSET_M) / COURT_LEN_M) * self.tw
        y_mid   = self.th / 2.0

        self.left_hoop_tv  = np.array([[x_left,  y_mid]], dtype=np.float32)
        self.right_hoop_tv = np.array([[x_right, y_mid]], dtype=np.float32)

        print(f"[HoopShadowDrawer] Initialized with tactical size=({self.tw},{self.th})")
        print(f"[HoopShadowDrawer] Left hoop tactical pos={self.left_hoop_tv}, Right hoop tactical pos={self.right_hoop_tv}")

    def _draw_marker(self, img, p, text):
        x, y = int(p[0]), int(p[1])
        if x <= 0 or y <= 0 or x >= img.shape[1] or y >= img.shape[0]:
            print(f"[HoopShadow] Skipped drawing {text}: out of frame coords=({x},{y})")
            return
        print(f"[HoopShadow] Drawing {text} at ({x},{y})")
        cv2.rectangle(img, (x - self.box//2, y - self.box//2),
                           (x + self.box//2, y + self.box//2), self.color, 2)
        cv2.putText(img, text, (x + self.box, y - self.box),
                    cv2.FONT_HERSHEY_SIMPLEX, self.label_scale, self.color, self.label_thickness, cv2.LINE_AA)

    def draw(self, frames, court_keypoints_per_frame):
        out = []
        for idx, (frame, kp_obj) in enumerate(zip(frames, court_keypoints_per_frame)):
            img = frame.copy()
            if kp_obj is None or kp_obj.xy is None or len(kp_obj.xy) == 0:
                print(f"[HoopShadow] Frame {idx}: no keypoints, skipping")
                out.append(img); continue

            frame_kps = kp_obj.xy.tolist()[0]
            valid_idx = [i for i, pt in enumerate(frame_kps) if pt[0] > 0 and pt[1] > 0]
            print(f"[HoopShadow] Frame {idx}: {len(valid_idx)} valid keypoints")

            if len(valid_idx) < 4:
                print(f"[HoopShadow] Frame {idx}: not enough keypoints for homography")
                out.append(img); continue

            source_points = np.array([frame_kps[i] for i in valid_idx], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_idx], dtype=np.float32)

            try:
                H = Homography(source_points, target_points)      # camera → tactical
                lh_img = H.inverse_transform_points(self.left_hoop_tv)[0]
                rh_img = H.inverse_transform_points(self.right_hoop_tv)[0]
                print(f"[HoopShadow] Frame {idx}: LH={lh_img}, RH={rh_img}")

                self._draw_marker(img, lh_img, "LH")
                self._draw_marker(img, rh_img, "RH")

            except Exception as e:
                print(f"[HoopShadow] Frame {idx}: homography failed ({e})")

            out.append(img)
        return out

