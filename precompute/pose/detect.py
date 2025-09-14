import os, json, cv2, sys
from precompute.helpers.frame_source import read_frame_at
from precompute.helpers.progress import ProgressLogger

def main():
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    video_path   = C.get("video_path")
    pose_cfg     = C["pose"]
    scoring_cfg  = C["scoring"]
    openpose_root= pose_cfg["openpose_root"]
    model_folder = pose_cfg["model_folder"]
    out_path     = pose_cfg["out_jsonl"]
    net_res      = pose_cfg["net_resolution"]
    window_len   = int(pose_cfg.get("window_len", 90))
    viz_dir      = pose_cfg.get("viz_dir", "output/pose_viz")
    VIS          = bool(pose_cfg.get("visualize", False))

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"[Pose] video_path missing or not found: {video_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if VIS: os.makedirs(viz_dir, exist_ok=True)

    sys.path.append(os.path.join(openpose_root, "build", "python"))
    from openpose import pyopenpose as op
    params = {
        "model_folder": model_folder,
        "model_pose": "BODY_25",
        "net_resolution": net_res,
        "render_pose": 0,
        "display": 0
    }
    opw = op.WrapperPython(); opw.configure(params); opw.start()

    events = []
    with open(scoring_cfg["out_jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if "trigger_idx" in rec:
                events.append((rec.get("event_id", len(events)+1), int(rec["trigger_idx"])))

    LOWER_IDX_LEFT  = (12,13,14)
    LOWER_IDX_RIGHT = (9,10,11)
    CONF_T = 0.20
    def has_lower(person):
        def vis(ids): return sum(float(person[i,2]) >= CONF_T for i in ids)
        return (vis(LOWER_IDX_LEFT) >= 2) or (vis(LOWER_IDX_RIGHT) >= 2)

    logger = ProgressLogger(prefix="Pose", log_every=50)

    with open(out_path, "w", encoding="utf-8") as jf:
        for ev_id, t in events:
            start = max(0, t - (window_len - 1))
            event_obj = {"event_id": ev_id, "frames": []}
            for i in range(start, t+1):
                img = read_frame_at(video_path, i)
                fname = f"frame_{i:06d}.png"
                if img is None:
                    event_obj["frames"].append({"frame": fname, "people": []})
                    logger.tick()
                    continue

                datum = op.Datum(); datum.cvInputData = img
                v = op.VectorDatum(); v.append(datum); opw.emplaceAndPop(v)

                people_out = []
                if datum.poseKeypoints is not None:
                    for person in datum.poseKeypoints:
                        if not has_lower(person): continue
                        flat=[]
                        for j in range(person.shape[0]):
                            x=float(person[j,0]); y=float(person[j,1]); c=float(person[j,2])
                            flat.extend([round(x,1), round(y,1), round(c,2)])
                            if VIS and c>=0.2:
                                cv2.circle(img,(int(round(x)),int(round(y))),2,(0,255,0),-1)
                        people_out.append({"p": flat})

                event_obj["frames"].append({"frame": fname, "people": people_out})
                if VIS:
                    cv2.imwrite(os.path.join(viz_dir, f"ev{ev_id}_{fname}"), img)
                logger.tick()

            jf.write(json.dumps(event_obj, separators=(",",":"))+"\n")

    logger.done(f"Output saved: {out_path}")

if __name__ == "__main__":
    main()

