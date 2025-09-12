# main.py
import json
import os

CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")
PRECOMPUTE_BASE = "../precompute"

from identify_shooter import IdentifyShooter
from helpers.load_jsonl import load_jsonl
from hoop_shadow_event import HoopShadowForEvent
from tactical_view_converter import TacticalViewConverter
from distance_to_hoop_drawer import DistanceToHoopDrawer
from height_estimator import HeightEstimator

from assign_team import TeamPrototype, HipColorAssigner

# --- 1/2/3-point differentiator (pure module, no CLI) ---
from differentiate_points import (
    build_frame_order,
    index_pose_by_frame,
    index_court_by_frame,
    decision_for_event,
)

# ------------------ CONFIG / PATHS ------------------
with open(CONFIG_PATH, "r") as f:
    C = json.load(f)

# I/O paths (resolve relative to ../precompute)
frames_dir_rel   = C.get("frames_dir")                       # e.g., "frames/006"
frames_dir_full  = os.path.join(PRECOMPUTE_BASE, frames_dir_rel)

scoring_jsonl    = os.path.join(PRECOMPUTE_BASE, C["scoring"]["out_jsonl"])
pose_jsonl_path  = os.path.join(PRECOMPUTE_BASE, C["pose"]["out_jsonl"])
court_jsonl_path = os.path.join(PRECOMPUTE_BASE, C["court"]["out_jsonl"])

# Optional: hoop-shadow debug jsonl path (cleared on start)
out_path = (C.get("hoop_shadow", {}) or {}).get("out_jsonl") \
           or os.path.join(PRECOMPUTE_BASE, "output", "hoop_shadow_points.jsonl")
if os.path.exists(out_path):
    os.remove(out_path)

# Where we’ll write final per-event summaries
summary_out_jsonl = os.path.join(PRECOMPUTE_BASE, "output", "game_logic_summary.jsonl")
os.makedirs(os.path.dirname(summary_out_jsonl), exist_ok=True)
# start fresh
if os.path.exists(summary_out_jsonl):
    os.remove(summary_out_jsonl)

# ------------------ INSTANTIATE ------------------
shooter = IdentifyShooter()
tvc = TacticalViewConverter(court_image_path="basketball_court.png")
shadow = HoopShadowForEvent(tvc, config_path="config.json", write_output=True)
estimator = HeightEstimator(
    config_path=CONFIG_PATH,
    require_vertical_ok_for_scale=False,  # set True if you want stricter scaling
    use_eye_ratio=True,
    eye_to_height_ratio=0.93,
    nose_to_height_ratio=0.96,
    eye_to_vertex_add_m=0.12,
    nose_to_vertex_add_m=0.10
)
drawer = DistanceToHoopDrawer(
    config_path="config.json",
    out_root=os.path.join(PRECOMPUTE_BASE, "output", "hoop_shadow_viz"),
    require_vertical_ok=False
)

TEAM_A = TeamPrototype(name="RED",   hue_deg=176.5, sat_mean=0.82, a_med=185, b_med=162)
TEAM_B = TeamPrototype(name="WHITE", hue_deg=154.0, sat_mean=0.05, a_med=133, b_med=124)
assigner = HipColorAssigner(frames_dir_full, TEAM_A, TEAM_B)

# ------------------ BUILD INDEXES ONCE ------------------
# differentiate_points expects frame *filenames* (e.g. "frame_001.png")
frame_order = build_frame_order(frames_dir_rel, precomp_base=PRECOMPUTE_BASE)
pose_idx    = index_pose_by_frame(pose_jsonl_path)
court_idx   = index_court_by_frame(court_jsonl_path)
print(f"[differentiate] frames={len(frame_order)} pose={len(pose_idx)} court={len(court_idx)}")

# ------------------ MAIN PIPELINE ------------------
def main():
    scoring_events = load_jsonl(scoring_jsonl)
    if not scoring_events:
        print(f"[WARN] No scoring events in: {scoring_jsonl}")
        return

    with open(summary_out_jsonl, "a", encoding="utf-8") as out_f:
        for ev in scoring_events:
            # 1) Shooter identification
            result = shooter.identify_shooter(ev)
            if result == -1:
                print("Shooter not identified — skipping rest of pipeline")
                continue

            print(result)

            shooter_frame = result.get("shooter_frame") or result.get("frame") or ev.get("frame")
            person_index  = result.get("person_index")

            # 2) Hoop shadow + distance viz
            shadow_rows = shadow.compute_for_event(ev)
            _written = drawer.draw_for_event(ev, shadow_rows=shadow_rows)

            # 3) Height estimation
            height_est = estimator.estimate_for_event(ev, shadow_rows, result)

            # 4) Team assignment (hip-color heuristic)
            res = assigner.assign_from_pose_jsonl(shooter_frame, int(person_index), pose_jsonl_path, debug=True)
            team_label, team_conf, dbg = res  # debug=True returns details
            for line in dbg:
                print("[assigner]", line)

            # 5) 1PT / 2PT / 3PT decision
            event_for_decision = {**ev, **result}  # ensure trigger_frame, shooter_frame, person_index present
            pt_res = decision_for_event(
                event_for_decision,
                frame_order,
                pose_idx,
                court_idx,
                ankle_dist_thresh_px=50.0,  # px to FT line to call 1PT
                min_ft_hits=1,              # need at least 2/3 frames near line
                frame_gap_3pt=49,           # >50 frames between trigger and shot -> 3PT
                debug=True
            )
            points_label  = pt_res["label"]
            points_reason = pt_res.get("reason")

            # 6) Final one-line summary for this event (print + write)
            summary = {
                "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
                "frame": shooter_frame,
                "person_index": int(person_index) if person_index is not None else None,
                "shadow_rows_written": len(shadow_rows),
                "height_estimate_m": height_est,   # keep units clear
                "team_label": team_label,
                "team_confidence": round(team_conf, 3),
                "points_label": points_label,      # "1PT" | "2PT" | "3PT" | "UNKNOWN"
                "points_reason": points_reason,    # e.g. "gap>50", "ft_hits>=2", "default"
            }
            print(summary)
            out_f.write(json.dumps(summary) + "\n")

if __name__ == "__main__":
    main()

