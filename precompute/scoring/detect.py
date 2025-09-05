import cv2, glob, os, sys, json

#load config
CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    C = json.load(f)

#paths from config
frames_dir = C["frames_dir"]
hoop_cfg = C["hoop"]
ball_cfg = C["ball"]
scoring_cfg = C["scoring"]
hoop_detections = hoop_cfg["out_jsonl"]
ball_detections = ball_cfg["out_jsonl"]
vis = scoring_cfg["visualize"]
out_path = scoring_cfg["out_jsonl"]
out_folder = "output"

# --- load frames ---
frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

# --- load hoop annotations ---
hoop_detect = {}
with open(hoop_detections, "r") as f:
    print(f"[DEBUG]: hoop file {hoop_detections}")
    for line in f:
        detection = json.loads(line)
        frame_name = detection["frame"]
        hoop_detect[frame_name] = detection["bbox"]

# --- load ball annotations ---
ball_detect = {}
with open(ball_detections, "r") as f:
    print(f"[DEBUG]: ball file {ball_detections}")
    for line in f:
        detection = json.loads(line)
        frame_name = detection["frame"]
        ball_detect[frame_name] = detection

print(f"[Scoring] Loaded {len(frame_paths)} frames")
print(f"[Scoring] Loaded {len(hoop_detect)} hoop detections, {len(ball_detect)} ball detections")

def to_topleft_xywh_from_center(cx, cy, w, h):
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return x, y, int(round(w)), int(round(h))

def compute_above_below_zones(bx, by, bw, bh, W, H):
    zone_h_a = int(round(1 * bh))
    zone_h_b = int(round(2 * bh))
    above = [bx, max(0, by - zone_h_a), bw, zone_h_a]
    below = [bx-10, min(H - 1, by + bh), bw+20, min(zone_h_b, H - (by + bh))]
    return above, below

def point_in_box(px, py, box):
    x, y, w, h = box
    return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)

# ----- simple state machine -----
state = "IDLE"                # IDLE -> ABOVE -> SIZE_CHANGED -> (detect BELOW) -> IDLE
inside_flag = False           # becomes True when ball enters above zone
ref_bw = None
ref_bh = None
frames_left = 0               # countdown window for transitions
WINDOW_AFTER_ABOVE = 10       # frames to see bbox size change after ABOVE
WINDOW_AFTER_SIZE = 5        # frames to see ball in BELOW after size change
SIZE_DELTA_PX = 7            # threshold for hoop bbox width/height change
scoring_event_count = 0

# --- main loop ---
with open(out_path, "w") as jf:
    for frame_path in frame_paths:
        if (vis):
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[WARN] Could not read image: {frame_path}")
                continue
            H, W = img.shape[:2]

        H = 1080
        W = 1920
        fname = os.path.basename(frame_path)

        # Initialize per-frame vars so later checks are safe
        cx = cy = bw = bh = None
        above = below = None
        bx_c = by_c = None
        have_hoop = False
        have_ball = False

        # hoop + zones
        if fname in hoop_detect:
            cx, cy, bw, bh = map(float, hoop_detect[fname])
            bx, by, bw, bh = to_topleft_xywh_from_center(cx, cy, bw, bh)
            have_hoop = True

            above, below = compute_above_below_zones(bx, by, bw, bh, W, H)
            ax, ay, aw, ah = above
            zx, zy, zw, zh = below

            if (vis):
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)  # hoop in red
                cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 1)  # above in green
                cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), (0, 255, 0), 1)  # below in green

        #ball
        det = ball_detect.get(fname)
        if det and det.get("coordinates") is not None:
            coords = det["coordinates"]
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                bx_c, by_c = int(round(coords[0])), int(round(coords[1]))
                have_ball = True

                if vis:
                    cv2.circle(img, (bx_c, by_c), 6, (0, 255, 0), -1)
                    score = det.get("score")
                    if score is not None:
                        cv2.putText(
                            img, f"{float(score):.2f}",
                            (bx_c + 8, by_c - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                            lineType=cv2.LINE_AA
                        )

        # ------ state updates ------
        if (cx is not None) and (bx_c is not None) and (above is not None) and (below is not None):
            in_above = point_in_box(bx_c, by_c, above)
            in_below = point_in_box(bx_c, by_c, below)

            if state == "IDLE":
                if in_above:
                    print("SCORING: in above")
                    state = "ABOVE"
                    inside_flag = True
                    ref_bw, ref_bh = bw, bh
                    frames_left = WINDOW_AFTER_ABOVE

            elif state == "ABOVE":
                frames_left -= 1
                # check size change
                print(f"{max(abs(bw - ref_bw), abs(bh - ref_bh))} pixels changed")
                if max(abs(bw - ref_bw), abs(bh - ref_bh)) >= SIZE_DELTA_PX:
                    state = "SIZE_CHANGED"
                    print("SCORING: size changed of hoop bbox")
                    frames_left = WINDOW_AFTER_SIZE
                elif frames_left <= 0:
                    # timeout -> reset
                    print("SCORING: reset")
                    state = "IDLE"
                    inside_flag = False
                    ref_bw = ref_bh = None

            elif state == "SIZE_CHANGED":
                frames_left -= 1
                if in_below:
                    print("SCORNG: in below")
                    scoring_event_count += 1
                    # --- write just frame + counter ---
                    event = {"frame": fname, "scoring_event_count": scoring_event_count}
                    jf.write(json.dumps(event) + "\n")

                    # reset
                    state = "IDLE"
                    inside_flag = False
                    ref_bw = ref_bh = None
                    frames_left = 0
                elif frames_left <= 0:
                    # didn't see the ball below in time -> reset
                    print("SCORING: reset")
                    state = "IDLE"
                    inside_flag = False
                    ref_bw = ref_bh = None

        if (vis):
            save_path = os.path.join(out_folder, fname)
            ok = cv2.imwrite(str(save_path), img)
            if not ok:
                print(f"Failed to save {save_path}")

print(f"[Scoring] Done. Scores detected: {scoring_event_count}")
print(f"[Scoring] Detections saved to {out_path}")
