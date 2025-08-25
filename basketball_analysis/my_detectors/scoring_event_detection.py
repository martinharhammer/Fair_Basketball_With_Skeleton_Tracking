# scoring_from_files.py
import os, json, cv2, datetime as dt

# --- helpers copied 1:1 ---
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

def run_scoring_from_files(
    frames_dir,
    hoop_annos_path,
    ball_annos_path,
    output_dir,
    output_name="scoring_events.jsonl",
    draw_debug=False,   # set True to save debug frames with rectangles/circles
):
    # --- paths / outputs ---
    os.makedirs(output_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dated_output_dir = os.path.join(output_dir, stamp)
    if draw_debug:
        os.makedirs(dated_output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # --- load frames (filenames only, like your script) ---
    frame_files = sorted([f for f in os.listdir(frames_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # --- load hoop annotations (bbox = [cx,cy,w,h]) ---
    hoop_annos = {}
    with open(hoop_annos_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ann = json.loads(line[line.find("{"):] if "{" in line else line)
            hoop_annos[ann["frame"]] = ann

    # --- load ball annotations ("coordinates":[x,y], optional "score") ---
    ball_annos = {}
    with open(ball_annos_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ann = json.loads(line[line.find("{"):] if "{" in line else line)
            ball_annos[ann["frame"]] = ann

    print(f"[INFO] Loaded {len(frame_files)} frames")
    print(f"[INFO] Loaded {len(hoop_annos)} hoop annos, {len(ball_annos)} ball annos")

    # ----- simple state machine (unchanged names/defaults) -----
    state = "IDLE"                # IDLE -> ABOVE -> SIZE_CHANGED -> (detect BELOW) -> IDLE
    inside_flag = False
    ref_bw = None
    ref_bh = None
    frames_left = 0
    WINDOW_AFTER_ABOVE = 10
    WINDOW_AFTER_SIZE  = 5
    SIZE_DELTA_PX      = 10
    scoring_event_count = 0

    # --- main loop ---
    with open(output_path, "w", encoding="utf-8") as events_file:
        for fname in frame_files:
            frame_path = os.path.join(frames_dir, fname)
            img = cv2.imread(frame_path)
            if img is None:
                continue
            H, W = img.shape[:2]

            # per-frame vars
            cx = cy = bw = bh = None
            above = below = None
            bx_c = by_c = None

            # hoop + zones
            if fname in hoop_annos and "bbox" in hoop_annos[fname]:
                cx, cy, bw, bh = map(float, hoop_annos[fname]["bbox"])
                bx, by, bw, bh = to_topleft_xywh_from_center(cx, cy, bw, bh)

                if draw_debug:
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)  # hoop in red
                above, below = compute_above_below_zones(bx, by, bw, bh, W, H)
                if draw_debug:
                    ax, ay, aw, ah = above
                    zx, zy, zw, zh = below
                    cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 1)  # above in green
                    cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), (0, 255, 0), 1)  # below in green

            # ball
            if fname in ball_annos and ball_annos[fname].get("coordinates") is not None:
                bx_c, by_c = ball_annos[fname]["coordinates"]
                bx_c, by_c = int(round(bx_c)), int(round(by_c))
                if draw_debug:
                    cv2.circle(img, (bx_c, by_c), 6, (0, 255, 0), -1)
                    if "score" in ball_annos[fname] and ball_annos[fname]["score"] is not None:
                        score = float(ball_annos[fname]["score"])
                        cv2.putText(
                            img, f"{score:.2f}", (bx_c + 8, by_c - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                            lineType=cv2.LINE_AA
                        )

            # ------ state updates (identical logic) ------
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
                        # write just frame + counter
                        event = {"frame": fname, "scoring_event_count": scoring_event_count}
                        events_file.write(json.dumps(event) + "\n")

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

            # optional debug frame dump
            if draw_debug:
                cv2.imwrite(os.path.join(dated_output_dir, fname), img)

    print(f"[INFO] Done. Scores detected: {scoring_event_count}")
    return scoring_event_count, output_path

