import os, json, cv2, torch, numpy as np
from PIL import Image
from omegaconf import OmegaConf
from contextlib import redirect_stdout, redirect_stderr
import io

from .hrnet import HRNet
from .preprocessing import ResizeWithEqualScale, SeqTransformCompose
from .postprocessor import WASBPostprocessor

from precompute.helpers.frame_source import FrameSource, sliding_windows
from precompute.helpers.progress import ProgressLogger


def main():
    # --------------- CONFIGURATION ---------------
    CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    video_path    = C.get("video_path")
    BALL          = C["ball"]
    OUTPUT_FOLDER = BALL["output_folder"]
    OUT_PATH      = BALL["out_jsonl"]
    HRNET_CFG     = BALL["hrnet_config"]
    WEIGHTS_PATH  = BALL["weights_path"]

    N_FRAMES  = int(BALL.get("n_frames", 3))
    DEBUG     = bool(BALL.get("debug", False))
    LOG_EVERY = int(BALL.get("log_every", 50))
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    # --------------- LOAD CONFIG ---------------
    cfg = OmegaConf.load(HRNET_CFG)

    # --------------- BUILD MODEL ---------------
    model = HRNet(cfg)
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # --------------- POSTPROCESSOR ---------------
    post = WASBPostprocessor(
        heatmap_threshold=float(BALL.get("heatmap_threshold", 0.2)),
        distance_threshold=int(BALL.get("distance_threshold", 5)),
        history_size=int(BALL.get("history_size", 3)),
    )

    # --------------- PREPROCESSOR ---------------
    resize = ResizeWithEqualScale(
        height=cfg['inp_height'],
        width=cfg['inp_width']
    )
    seq_transform = SeqTransformCompose(frame_transform=resize)

    # --------------- FRAME SOURCE ---------------
    src = FrameSource(video_path=video_path)

    # no-op affine mats
    import torch as _torch
    eye2x3 = _torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=_torch.float32)
    affine_mats = {0: eye2x3.unsqueeze(0).unsqueeze(0).repeat(1, N_FRAMES, 1, 1)}

    total = (src.count - N_FRAMES + 1) if (src.count is not None) else None
    logger = ProgressLogger("Ball", total=total, log_every=LOG_EVERY)

    # --------------- INFERENCE LOOP ---------------
    with open(OUT_PATH, "w") as jf:
        for center_idx, names, frames_bgr in sliding_windows(iter(src), window=N_FRAMES, step=1):
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
            inp = seq_transform(pil_frames).unsqueeze(0).to(DEVICE)

            with _torch.no_grad():
                preds = model(inp)

            if DEBUG:
                results = post.run(preds, affine_mats)
            else:
                _sink_out, _sink_err = io.StringIO(), io.StringIO()
                with redirect_stdout(_sink_out), redirect_stderr(_sink_err):
                    results = post.run(preds, affine_mats)

            scale = list(preds.keys())[0]
            mid = post.mid
            dets = results[0][mid][scale]

            xy, score = (None, None)
            if len(dets['xys']) > 0:
                xy, score = dets['xys'][0], dets['scores'][0]

            frame_name = names[mid]
            H, W = frames_bgr[mid].shape[:2]
            inp_h, inp_w = cfg['inp_height'], cfg['inp_width']

            if xy is not None:
                xy_scaled = [float(xy[0] * W / inp_w), float(xy[1] * H / inp_h)]
            else:
                xy_scaled = None

            record = {
                "frame": frame_name,
                "coordinates": xy_scaled,
                "score": float(score) if score is not None else None
            }
            jf.write(json.dumps(record) + "\n")

            logger.tick()

    logger.done(f"Output saved: {OUT_PATH}")

if __name__ == "__main__":
    main()

