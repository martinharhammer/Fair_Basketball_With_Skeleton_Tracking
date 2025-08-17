import cv2
import json
import os
import datetime as dt

# --- paths ---
frames_dir = "001/raw_frames"
hoop_annos_path = "001/hoop/detections.jsonl"   # "bbox": [cx, cy, w, h] (center format)
ball_annos_path = "001/ball/detections.jsonl"      # "coordinates": [x, y]
output_dir = "output/scoring_event_detection"

stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dated_output_dir = os.path.join(output_dir, stamp)
#os.makedirs(dated_output_dir, exist_ok=True)
print(f"[INFO] Saving frames to: {output_dir}")

# --- load frames ---
frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# --- load hoop annotations ---
hoop_annos = {}
with open(hoop_annos_path, "r") as f:
    for line in f:
        ann = json.loads(line)
        hoop_annos[ann["frame"]] = ann

# --- load ball annotations ---
ball_annos = {}
with open(ball_annos_path, "r") as f:
    for line in f:
        ann = json.loads(line)
        ball_annos[ann["frame"]] = ann

print(f"[INFO] Loaded {len(frame_files)} frames")
print(f"[INFO] Loaded {len(hoop_annos)} hoop annos, {len(ball_annos)} ball annos")

def to_topleft_xywh_from_center(cx, cy, w, h):
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return x, y, int(round(w)), int(round(h))

def compute_above_below_zones(bx, by, bw, bh, W, H):
    zone_h = int(round(0.2 * bh))
    above = [bx, max(0, by - zone_h), bw, zone_h]
    below = [bx, min(H - 1, by + bh), bw, min(zone_h, H - (by + bh))]
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
SIZE_DELTA_PX = 10            # threshold for hoop bbox width/height change
scoring_event_count = 0

# --- main loop ---
for fname in frame_files:
    frame_path = os.path.join(frames_dir, fname)
    img = cv2.imread(frame_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    # Initialize per-frame vars so later checks are safe
    cx = cy = bw = bh = None
    above = below = None
    bx_c = by_c = None

    # hoop + zones
    if fname in hoop_annos and "bbox" in hoop_annos[fname]:
        cx, cy, bw, bh = map(float, hoop_annos[fname]["bbox"])
        bx, by, bw, bh = to_topleft_xywh_from_center(cx, cy, bw, bh)

        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)  # hoop in red
        above, below = compute_above_below_zones(bx, by, bw, bh, W, H)
        ax, ay, aw, ah = above
        zx, zy, zw, zh = below
        cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 1)  # above in green
        cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), (0, 255, 0), 1)  # below in green

    # ball
    if fname in ball_annos and ball_annos[fname].get("coordinates") is not None:
        bx_c, by_c = ball_annos[fname]["coordinates"]
        bx_c, by_c = int(round(bx_c)), int(round(by_c))
        cv2.circle(img, (bx_c, by_c), 6, (0, 255, 0), -1)   # blue filled
        if "score" in ball_annos[fname] and ball_annos[fname]["score"] is not None:
                score = float(ball_annos[fname]["score"])
                cv2.putText(
                    img, f"{score:.2f}", (bx_c + 8, by_c - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    lineType=cv2.LINE_AA
                )

    # ------ state updates ------
    if (cx is not None) and (bx_c is not None) and (above is not None) and (below is not None):
        in_above = point_in_box(bx_c, by_c, above)
        in_below = point_in_box(bx_c, by_c, below)

        if state == "IDLE":
            if in_above:
                state = "ABOVE"
                inside_flag = True
                ref_bw, ref_bh = bw, bh
                frames_left = WINDOW_AFTER_ABOVE

        elif state == "ABOVE":
            frames_left -= 1
            # check size change
            if max(abs(bw - ref_bw), abs(bh - ref_bh)) >= SIZE_DELTA_PX:
                state = "SIZE_CHANGED"
                frames_left = WINDOW_AFTER_SIZE
            elif frames_left <= 0:
                # timeout -> reset
                state = "IDLE"
                inside_flag = False
                ref_bw = ref_bh = None

        elif state == "SIZE_CHANGED":
            frames_left -= 1
            if in_below:
                scoring_event_count += 1
                # reset
                state = "IDLE"
                inside_flag = False
                ref_bw = ref_bh = None
                frames_left = 0
            elif frames_left <= 0:
                # didn't see the ball below in time -> reset
                state = "IDLE"
                inside_flag = False
                ref_bw = ref_bh = None

    # HUD: top-right counter
    text = f"Scoring event detected: {scoring_event_count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x0 = W - tw - 16
    y0 = 16 + th
    cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, lineType=cv2.LINE_AA)
        

    # save
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, img)

print("[INFO] Done.")

