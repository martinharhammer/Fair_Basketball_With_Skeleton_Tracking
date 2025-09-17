from pathlib import Path
import json, os

from .utils.config import load_config
from .utils.frames import frame_name_to_index, hms_from_frame 
from .utils.scoring import points_to_value
from .utils.io import collect_frames_from_jsonls, append_jsonl, write_json, ensure_dir, load_jsonl
from .utils.geom import get_center_of_bbox, get_bbox_width, measure_distance, get_foot_position 
from .utils.video import get_video_fps_strict

from .identify_shooter import IdentifyShooter
from .assign_team import HoopSideTeamAssigner
from .height_estimation.hoop_shadow import HoopShadowForEvent
from .homography.tactical_view_converter import TacticalViewConverter
from .height_estimation.estimate_height import HeightEstimator
from . import compute_fair_score as cns

from .differentiate_points import (
    index_pose_by_frame,
    index_court_by_frame,
    decision_for_event,
)
# ------------------ CONFIG / PATHS ------------------
CONFIG_ENV = os.environ.get("GATHER_CONFIG", "config.json")
C, resolve = load_config(CONFIG_ENV)

video_path       = resolve(C.get("video_path"))
frames_dir_full  = resolve(C.get("frames_dir")) if C.get("frames_dir") else None
scoring_jsonl    = resolve(C["scoring"]["out_jsonl"])
pose_jsonl_path  = resolve(C["pose"]["out_jsonl"])
court_jsonl_path = resolve(C["court"]["out_jsonl"])

# outputs (strict from config)
summary_out_jsonl = resolve(C["game_logic"]["summary_out_jsonl"])
score_input_json  = resolve(C["game_logic"]["score_input_json"])
final_score_txt   = resolve(C["game_logic"]["fair_score"])
hoop_shadow_out   = resolve(C["hoop_shadow"]["out_jsonl"])

# ------------------ INSTANTIATE ------------------
CONFIG_ABS = str(Path(CONFIG_ENV).resolve())

shooter = IdentifyShooter()
tvc = TacticalViewConverter(court_image_path=resolve("basketball_court.png"))
shadow = HoopShadowForEvent(tvc, config_path=CONFIG_ABS)
estimator = HeightEstimator(
    config_path=CONFIG_ABS,
    require_vertical_ok_for_scale=False,
    use_eye_ratio=True,
    eye_to_height_ratio=0.93,
    nose_to_height_ratio=0.96,
    eye_to_vertex_add_m=0.12,
    nose_to_vertex_add_m=0.10,
)
hoop_side_assigner = HoopSideTeamAssigner(config_path=CONFIG_ABS)

ensure_dir(os.path.dirname(summary_out_jsonl))
ensure_dir(os.path.dirname(score_input_json))

# ------------------ FRAME ORDER FROM JSONLs ------------------
frame_order = collect_frames_from_jsonls(pose_jsonl_path, court_jsonl_path, scoring_jsonl)
pose_idx    = index_pose_by_frame(pose_jsonl_path)
court_idx   = index_court_by_frame(court_jsonl_path)

# ------------------ MAIN PIPELINE ------------------
def main():
    scoring_events = load_jsonl(scoring_jsonl)
    if not scoring_events:
        print(f"[WARN] No scoring events in: {scoring_jsonl}")
        return

    # ---- scoring output setup ----
    if not video_path:
        raise RuntimeError("[CONFIG] video_path is required to compute timestamps reliably.")
    fps = get_video_fps_strict(video_path)  # read from actual video (most reliable)

    score_rows = []

    with open(summary_out_jsonl, "w", encoding="utf-8") as out_f:
        for ev in scoring_events:
            result = shooter.identify_shooter(ev)

            no_shooter = (
                result is None
                or (isinstance(result, int) and result == -1)
                or (isinstance(result, dict) and (result.get("person_index") is None or result.get("shooter_frame") is None))
            )

            if no_shooter:
                frame_for_ts = ev.get("frame") or ev.get("trigger_frame")
                ts = None
                if frame_for_ts:
                    try:
                        ts = hms_from_frame(frame_name_to_index(frame_for_ts), fps)
                    except Exception:
                        ts = None

                out_f.write(json.dumps({
                    "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
                    "frame": frame_for_ts,
                    "shooter": None,
                    "person_index": None,
                    "shadow_rows_written": 0,
                    "height_estimate_m": None,
                    "team_label": "UNKNOWN",
                    "team_confidence": 0.0,
                    "points_label": "UNKNOWN",
                    "points_reason": "no_shooter"
                }) + "\n")

                score_rows.append({
                    "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
                    "timestamp": ts,
                    "frame": frame_for_ts,
                    "points": 0,
                    "points_label": "UNKNOWN",
                    "height_m": None,
                    "team": "UNKNOWN"
                })
                continue

            shooter_frame = result.get("shooter_frame") or result.get("frame") or ev.get("frame")

            shadow_rows = shadow.compute_for_event(ev)

            height_est = estimator.estimate_for_event(ev, shadow_rows, result)
            assigned_team = hoop_side_assigner.assign(trigger_frame=shooter_frame)

            event_for_decision = {**ev, **result}
            pt_res = decision_for_event(
                event_for_decision,
                frame_order,
                pose_idx,
                court_idx,
                ankle_dist_thresh_px=50.0,
                min_ft_hits=1,
                frame_gap_3pt=49,
                debug=False
            )
            points_label  = pt_res["label"]
            points_reason = pt_res.get("reason")

            trigger_frame = ev.get("frame") or ev.get("trigger_frame")
            frame_idx = frame_name_to_index(trigger_frame)

            score_rows.append({
                "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
                "timestamp": hms_from_frame(frame_idx, fps),   # "HH:MM:SS.mmm"
                "frame": trigger_frame,                        # "frame_XXXXX.png"
                "points": points_to_value(points_label),       # 0/1/2/3
                "points_label": points_label,
                "height_m": height_est,                        # float or null
                "team": assigned_team
            })

            summary = {
                "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
                "frame": shooter_frame,
                "shooter": True,
                "shadow_rows_written": len(shadow_rows),
                "height_estimate_m": height_est,
                "team_label": assigned_team,
                "points_label": points_label,
                "points_reason": points_reason,
            }
            out_f.write(json.dumps(summary) + "\n")

    # ---- write the compact scoring input JSON (already in loop order) ----
    with open(score_input_json, "w", encoding="utf-8") as f:
        json.dump({"events": score_rows}, f, indent=2, ensure_ascii=False)
    #print(f"[Point] Output saved: {score_input_json} (events={len(score_rows)})")

    ensure_dir(os.path.dirname(final_score_txt))
    cns.JSON_FILE = score_input_json
    cns.OUTPUT_TXT = final_score_txt
    cns.main()

if __name__ == "__main__":
    main()

