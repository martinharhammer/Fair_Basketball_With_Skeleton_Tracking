import os, json, cv2

RED   = (0, 0, 255)
GREEN = (0, 255, 0)
BALL  = (0, 255, 0)

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_per_frame(path):
    by_idx = {}
    if not os.path.exists(path):
        return by_idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            idx = obj.get("idx")
            if idx is None:
                frm = obj.get("frame")
                if isinstance(frm, str) and frm.startswith("frame_") and frm.endswith(".png"):
                    try:
                        idx = int(frm[6:12])
                    except Exception:
                        idx = None
            if idx is not None:
                by_idx[int(idx)] = obj
    return by_idx

def draw_rect_xywh(img, xywh, color, thickness=1):
    x, y, w, h = map(int, xywh)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)

def draw_circle(img, x, y, color, r=5):
    cv2.circle(img, (int(x), int(y)), r, color, thickness=-1, lineType=cv2.LINE_AA)

def put_text_top_center(img, text, y_margin=30, scale=1.0, color=(255,255,255), thickness=2):
    H, W = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (W - tw) // 2
    y = y_margin + th  # putText uses baseline at y; add text height to margin
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    C = load_config(CONFIG_PATH)

    video_path = C["video_path"]
    pf_jsonl   = C["viz"]["scoring"]["frames_out_jsonl"]
    out_dir    = C["viz"]["scoring"]["viz_path"]
    os.makedirs(out_dir, exist_ok=True)
    out_video  = os.path.join(out_dir, "scoring_overlay_progress.mp4")

    per_frame = load_per_frame(pf_jsonl)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    passed_above = False
    passed_size  = False
    passed_below = False

    event_counter = 0
    event_counted = False

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rec = per_frame.get(idx)
        if rec:
            state = rec.get("state") or "IDLE"
            flags = rec.get("flags") or {}
            in_below = bool(flags.get("in_below", False))

            tmp_above = passed_above
            tmp_size  = passed_size
            tmp_below = passed_below

            if state == "ABOVE":
                tmp_above = True
            elif state == "SIZE_CHANGED":
                tmp_above = True
                tmp_size  = True

            if in_below:
                tmp_below = True

            col_above = GREEN if tmp_above else RED
            col_hoop  = GREEN if tmp_size  else RED
            col_below = GREEN if tmp_below else RED

            hoop = rec.get("hoop") or {}
            xywh = hoop.get("xywh")
            if xywh:
                draw_rect_xywh(frame, xywh, col_hoop, 1)

            zones = rec.get("zones") or {}
            above = zones.get("above_xywh")
            below = zones.get("below_xywh")
            if above: draw_rect_xywh(frame, above, col_above, 1)
            if below: draw_rect_xywh(frame, below, col_below, 1)

            ball = rec.get("ball")
            if ball and (ball.get("x") is not None) and (ball.get("y") is not None):
                draw_circle(frame, ball["x"], ball["y"], BALL, r=5)

            if (tmp_size and tmp_below) and not event_counted:
                event_counter += 1
                event_counted = True

            put_text_top_center(frame, f"Scoring Events: {event_counter}", y_margin=30, scale=1.0)

            if state == "IDLE":
                passed_above = False
                passed_size  = False
                passed_below = False
                event_counted = False
            else:
                passed_above = tmp_above
                passed_size  = tmp_size
                passed_below = tmp_below
        else:
            put_text_top_center(frame, f"Scoring Events: {event_counter}", y_margin=30, scale=1.0)

        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    print(f"Wrote: {out_video}")

if __name__ == "__main__":
    main()

