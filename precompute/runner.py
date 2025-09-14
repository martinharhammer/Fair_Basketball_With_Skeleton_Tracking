# precompute/run.py
import os, json, subprocess, sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../precompute
REPO_ROOT = PKG_DIR.parent                         # repo root
DEFAULT_CONFIG = str((REPO_ROOT / "config.json").resolve())  # <— root config
CONFIG_PATH = os.environ.get("GATHER_CONFIG", DEFAULT_CONFIG)

def run_step(mod: str, tag: str, extra_env: dict | None = None):
    if not mod:
        return
    print(f"[RUN] {tag}: {mod} (subprocess)")
    env = os.environ.copy()
    env["GATHER_CONFIG"] = CONFIG_PATH

    # Ensure the repo is importable as a package root
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    if extra_env:
        env.update(extra_env)

    subprocess.run(
        [sys.executable, "-m", mod],
        check=True,
        cwd=str(REPO_ROOT),   # <— run from repo root so config-relative paths work
        env=env,
    )

def main():
    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    src_key = "video_path" if C.get("video_path") else "frames_dir"
    print(f"[PRECOMPUTE] Using {src_key}: {C.get(src_key)}")
    print(f"[CONFIG] {CONFIG_PATH}")

    steps = [
        ("ball",    "Ball"),
        ("hoop",    "Hoop"),
        ("court",   "Court"),
        ("scoring", "Scoring"),
        ("pose",    "Pose"),
    ]
    for key, tag in steps:
        spec = C.get(key, {}).get("module")
        if spec:
            run_step(spec, tag)

if __name__ == "__main__":
    main()

