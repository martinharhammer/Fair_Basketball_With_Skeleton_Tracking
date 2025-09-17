import os, json
from precompute.helpers.frame_source import FrameSource
from precompute.helpers.progress import ProgressLogger

def topleft_from_center(cx, cy, w, h):
    x = int(round(cx - w/2.0)); y = int(round(cy - h/2.0))
    return x, y, int(round(w)), int(round(h))

def zones(bx, by, bw, bh, W, H):
    za = int(round(1*bh)); zb = int(round(2*bh))
    above = [bx, max(0, by - za), bw, za]
    below = [bx-10, min(H-1, by + bh), bw+20, min(zb, H-(by+bh))]
    return above, below

def pip(px, py, box):
    x,y,w,h = box
    return (x <= px <= x+w) and (y <= py <= y+h)

def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    video_path = C.get("video_path")
    hoop_cfg   = C["hoop"]
    ball_cfg   = C["ball"]
    scoring_cfg= C["scoring"]

    hoop_jsonl = hoop_cfg["out_jsonl"]
    ball_jsonl = ball_cfg["out_jsonl"]
    out_path   = scoring_cfg["out_jsonl"]

    VIZ_SCORING = bool(C["viz"]["scoring"]["visualize"])
    frames_out_path = C["viz"]["scoring"]["frames_out_jsonl"]

    hoop_det = {}
    with open(hoop_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); hoop_det[d["frame"]] = d["bbox"]

    ball_det = {}
    with open(ball_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); ball_det[d["frame"]] = d

    state = "IDLE"; ref_bw = ref_bh = None; frames_left = 0
    WINDOW_AFTER_ABOVE = 10; WINDOW_AFTER_SIZE = 5; SIZE_DELTA_PX = 3
    ev_count = 0

    src = FrameSource(video_path=video_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if VIZ_SCORING:
        os.makedirs(os.path.dirname(frames_out_path) or ".", exist_ok=True)

    logger = ProgressLogger(prefix="Scoring", total=src.count or None, log_every=50)

    jf_frames = open(frames_out_path, "w", encoding="utf-8") if VIZ_SCORING else None
    with open(out_path, "w", encoding="utf-8") as jf:
        for idx, fname, img in src:
            H, W = (img.shape[0], img.shape[1]) if img is not None else (1080, 1920)
            cx=cy=bw=bh=None; above=below=None; bx_c=by_c=None
            in_above=False; in_below=False

            if fname in hoop_det:
                cx,cy,bw,bh = map(float, hoop_det[fname])
                bx,by,bw,bh = topleft_from_center(cx,cy,bw,bh)
                above, below = zones(bx,by,bw,bh,W,H)

            det = ball_det.get(fname)
            if det and det.get("coordinates") is not None:
                coords = det["coordinates"]
                if isinstance(coords,(list,tuple)) and len(coords)==2:
                    bx_c, by_c = int(round(coords[0])), int(round(coords[1]))

            if (cx is not None) and (bx_c is not None) and (above is not None) and (below is not None):
                in_above = pip(bx_c, by_c, above)
                in_below = pip(bx_c, by_c, below)

                if state == "IDLE":
                    if in_above:
                        state = "ABOVE"
                        ref_bw, ref_bh = bw, bh
                        frames_left = WINDOW_AFTER_ABOVE

                elif state == "ABOVE":
                    frames_left -= 1
                    if max(abs(bw - ref_bw), abs(bh - ref_bh)) >= SIZE_DELTA_PX:
                        state = "SIZE_CHANGED"
                        frames_left = WINDOW_AFTER_SIZE
                    elif frames_left <= 0:
                        state = "IDLE"; ref_bw = ref_bh = None

                elif state == "SIZE_CHANGED":
                    frames_left -= 1
                    if in_below:
                        ev_count += 1
                        jf.write(json.dumps({
                            "event_id": ev_count,
                            "frame": fname,
                            "trigger_idx": idx
                        })+"\n")
                        state = "IDLE"; ref_bw = ref_bh = None; frames_left = 0
                    elif frames_left <= 0:
                        state = "IDLE"; ref_bw = ref_bh = None

            if VIZ_SCORING and jf_frames is not None:
                frame_record = {
                    "frame": fname,
                    "idx": idx,
                    "state": state,
                    "flags": {"in_above": bool(in_above), "in_below": bool(in_below)},
                    "hoop": None,
                    "zones": None,
                    "ball": None
                }
                if cx is not None:
                    frame_record["hoop"] = {"xywh": [bx, by, bw, bh]}
                if above is not None and below is not None:
                    frame_record["zones"] = {
                        "above_xywh": [int(above[0]), int(above[1]), int(above[2]), int(above[3])],
                        "below_xywh": [int(below[0]), int(below[1]), int(below[2]), int(below[3])]
                    }
                if (bx_c is not None) and (by_c is not None):
                    frame_record["ball"] = {"x": int(bx_c), "y": int(by_c)}

                jf_frames.write(json.dumps(frame_record) + "\n")

            logger.tick()

    if jf_frames is not None:
        jf_frames.close()

    msg = f"Event log: {out_path}"
    if VIZ_SCORING:
        msg += f" | Per-frame log: {frames_out_path}"
    logger.done(msg)

if __name__ == "__main__":
    main()

