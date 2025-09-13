import os, json, cv2
from precompute.helpers.frame_source import FrameSource

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
    VIS        = bool(scoring_cfg.get("visualize", False))
    out_folder = scoring_cfg.get("viz_dir", "output")

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
    os.makedirs(out_folder, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as jf:
        for idx, fname, img in src:
            H, W = (img.shape[0], img.shape[1]) if img is not None else (1080, 1920)
            cx=cy=bw=bh=None; above=below=None; bx_c=by_c=None

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

            if VIS and img is not None:
                if above:  cv2.rectangle(img, (above[0],above[1]), (above[0]+above[2], above[1]+above[3]), (0,255,0),1)
                if below:  cv2.rectangle(img, (below[0],below[1]), (below[0]+below[2], below[1]+below[3]), (0,255,0),1)
                cv2.imwrite(os.path.join(out_folder, fname), img)

    print(f"[Scoring] Events: {ev_count} → {out_path}")

if __name__ == "__main__":
    main()

