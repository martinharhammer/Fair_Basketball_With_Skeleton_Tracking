import json
import os
CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")

from identify_shooter import IdentifyShooter
from helpers.load_jsonl import load_jsonl
from hoop_shadow_event import HoopShadowForEvent
from tactical_view_converter import TacticalViewConverter
from distance_to_hoop_drawer import DistanceToHoopDrawer
from height_estimator import HeightEstimator

from assign_team import TeamPrototype, HipColorAssigner

with open (CONFIG_PATH, "r",) as f:
    C = json.load(f)

out_path = (C.get("hoop_shadow", {}) or {}).get("out_jsonl") \
           or "../precompute/output/hoop_shadow_points.jsonl"

if os.path.exists(out_path):
    os.remove(out_path)

scoring_events = load_jsonl(C["scoring"]["out_jsonl"])

shooter = IdentifyShooter()
tvc = TacticalViewConverter(court_image_path="basketball_court.png")
shadow = HoopShadowForEvent(tvc, config_path="config.json", write_output=True)
estimator = HeightEstimator(config_path=CONFIG_PATH,
                            require_vertical_ok_for_scale=False,  # or True, your call
                            use_eye_ratio=True,                    # ratio model (0.93/0.96)
                            eye_to_height_ratio=0.93,
                            nose_to_height_ratio=0.96,
                            eye_to_vertex_add_m=0.12,
                            nose_to_vertex_add_m=0.10)

drawer  = DistanceToHoopDrawer(config_path="config.json",
                               out_root="../precompute/output/hoop_shadow_viz",
                               require_vertical_ok=False)

frames_dir = C.get("frames_dir") or "../precompute/frames/raw_frames"
pose_jsonl = (C.get("pose", {}) or {}).get("out_jsonl")

TEAM_A = TeamPrototype(name="RED", hue_deg=176.5, sat_mean=0.82, a_med=185, b_med=162)
TEAM_B = TeamPrototype(name="WHITE", hue_deg=154.0, sat_mean=0.05, a_med=133, b_med=124)
assigner = HipColorAssigner(frames_dir, TEAM_A, TEAM_B)

def main():

    for ev in scoring_events:
        result = shooter.identify_shooter(ev)

        if (result == -1):
            print("Shooter not identified skipping rest of pipeline") 
            continue 

        print(result)

        shooter_frame = result.get("shooter_frame") or result.get("frame") or ev.get("frame")
        person_index  = result.get("person_index")

        shadow_rows = shadow.compute_for_event(ev)

        written = drawer.draw_for_event(ev, shadow_rows=shadow_rows)

        height_est = estimator.estimate_for_event(ev, shadow_rows, result)

        res = assigner.assign_from_pose_jsonl(shooter_frame, int(person_index), pose_jsonl, debug=True)
        team_label, team_conf, dbg = res  # because debug=True returns 3-tuple
        for line in dbg:
            print("[assigner]", line)

        print({
            "event_id": ev.get("event_id") or ev.get("scoring_event_count"),
            "frame": shooter_frame,
            "shadow_rows_written": len(shadow_rows),
            "height_estimate": height_est,
            "team_label": team_label,
            "team_confidence": round(team_conf, 3),
        })

if __name__ == "__main__":
    main()
