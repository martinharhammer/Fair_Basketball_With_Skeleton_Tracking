import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

#folder_path = pathlib.Path(__file__).parent.resolve()
#sys.path.append(os.path.join(folder_path,"../"))
from ..utils.geom import get_foot_position, measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),

            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

    LEFT_IDX  = [0, 1, 2, 3, 4, 5, 8, 9]
    RIGHT_IDX = [10,11,12,13,14,15,16,17]
    SWAP_MAP = {0:10, 1:11, 2:12, 3:13, 4:14, 5:15, 8:16, 9:17,
                10:0, 11:1, 12:2, 13:3, 14:4, 15:5, 16:8, 17:9}

    # ...

    def _maybe_flip_sides(self, frame_keypoints_xy, image_width=None, margin_px=0):
        """
        Decide if sides are flipped; return (xy_swapped, did_flip).
        frame_keypoints_xy: array-like shape (18,2) with (0,0) = missing.
        """
        import numpy as np
        kps = np.array(frame_keypoints_xy, dtype=float).copy()

        # midline from KP 6 & 7 if present, else fallback to half width
        mids = [kps[m,0] for m in (6,7) if kps[m,0] > 0 and kps[m,1] > 0]
        if mids:
            mid_x = float(np.mean(mids))
        else:
            if image_width is None:
                x_max = float(np.max(kps[:,0])) if np.isfinite(np.max(kps[:,0])) else 0.0
                image_width = x_max if x_max > 0 else 2.0
            mid_x = 0.5 * image_width

        # vote: how many "left" indices lie to the right of midline?
        left_detected = [i for i in self.LEFT_IDX if kps[i,0] > 0 and kps[i,1] > 0]
        if not left_detected:
            return kps, False

        left_on_right = sum(1 for i in left_detected if kps[i,0] > mid_x + margin_px)
        ratio_on_right = left_on_right / max(1, len(left_detected))

        if ratio_on_right > 0.5:
            kps_swapped = kps.copy()
            for i in self.LEFT_IDX + self.RIGHT_IDX:
                j = self.SWAP_MAP[i]
                kps_swapped[i] = kps[j]
            print(f"[Flip] Frame: flipping sides (ratio_on_right={ratio_on_right:.2f})")
            return kps_swapped, True

        return kps, False

    def _apply_flip_to_keypoints_obj(self, keypoints_obj, did_flip):
        """If did_flip, swap indices in both .xy and .xyn in-place (works for torch tensors or numpy)."""
        if not did_flip:
            return

        for arr_name in ("xy", "xyn"):
            arr = getattr(keypoints_obj, arr_name)[0]  # shape (18, 2); torch.Tensor in Ultralytics

            # Make a same-type copy
            if hasattr(arr, "clone"):            # torch.Tensor
                swapped = arr.clone()
                for i in self.LEFT_IDX + self.RIGHT_IDX:
                    swapped[i] = arr[self.SWAP_MAP[i]]
                arr[:] = swapped
            else:                                # numpy fallback
                swapped = arr.copy()
                for i in self.LEFT_IDX + self.RIGHT_IDX:
                    swapped[i] = arr[self.SWAP_MAP[i]]
                arr[:] = swapped
        print("[Flip] Applied flip to keypoints_obj")

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, kp in enumerate(keypoints_list):
            if kp is None:
                print(f"[Validate] Frame {frame_idx}: no keypoints object")
                continue

            xy = kp.xy.tolist()[0]
            _, did_flip = self._maybe_flip_sides(xy)
            self._apply_flip_to_keypoints_obj(kp, did_flip)

            frame_keypoints = kp.xy.tolist()[0]
            detected_indices = [i for i, pt in enumerate(frame_keypoints) if pt[0] > 0 and pt[1] > 0]
            print(f"[Validate] Frame {frame_idx}: {len(detected_indices)} detected keypoints")

            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []
            for i in detected_indices:
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')
                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:
                        print(f"[Validate] Frame {frame_idx}: keypoint {i} invalid (error={error:.2f})")
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        tactical_player_positions = []

        for frame_idx, (kp, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}

            xy = kp.xy.tolist()[0]
            _, did_flip = self._maybe_flip_sides(xy)
            self._apply_flip_to_keypoints_obj(kp, did_flip)

            frame_keypoints = kp.xy.tolist()[0]

            if frame_keypoints is None or len(frame_keypoints) == 0:
                print(f"[Transform] Frame {frame_idx}: no keypoints")
                tactical_player_positions.append(tactical_positions)
                continue

            valid_indices = [i for i, pt in enumerate(frame_keypoints) if pt[0] > 0 and pt[1] > 0]
            print(f"[Transform] Frame {frame_idx}: {len(valid_indices)} valid court keypoints, {len(frame_tracks)} players")

            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            detected_keypoints = frame_keypoints
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)

            try:
                homography = Homography(source_points, target_points)

                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    tactical_position = homography.transform_points(player_position)

                    if (0 <= tactical_position[0][0] <= self.width and
                        0 <= tactical_position[0][1] <= self.height):
                        tactical_positions[player_id] = tactical_position[0].tolist()
                        print(f"[Transform] Frame {frame_idx}: player {player_id} → {tactical_positions[player_id]}")
                    else:
                        print(f"[Transform] Frame {frame_idx}: player {player_id} out of bounds")

            except (ValueError, cv2.error) as e:
                print(f"[Transform] Frame {frame_idx}: homography failed ({e})")

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions

