# precompute_video.py
import os, sys, json, subprocess

CONFIG_PATH = os.environ.get("GATHER_CONFIG", "config.json")

def run_script(script_path: str, tag: str):
    print(f"[RUN] {tag}: {script_path}")
    # inherit env; use the same interpreter
    subprocess.run([sys.executable, script_path], check=True)

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    src_key = "video_path" if C.get("video_path") else "frames_dir"
    print(f"[PRECOMPUTE] Using {src_key}: {C.get(src_key)}")

    ball_script    = C.get("ball", {}).get("script")
    hoop_script    = C.get("hoop", {}).get("script")
    court_script   = C.get("court", {}).get("script")
    scoring_script = C.get("scoring", {}).get("script")
    pose_script    = C.get("pose",  {}).get("script")

    if ball_script:    run_script(ball_script,   "Ball")
    if hoop_script:    run_script(hoop_script,   "Hoop")
    if court_script:   run_script(court_script,  "Court")
    if scoring_script: run_script(scoring_script,"Scoring")
    if pose_script:    run_script(pose_script,   "Pose")

if __name__ == "__main__":
    main()

