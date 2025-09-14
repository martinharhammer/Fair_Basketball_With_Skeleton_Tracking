# precompute/run.py
import os, json, subprocess, sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DEFAULT_CONFIG = str((REPO_ROOT / "config.json").resolve())
CONFIG_PATH = os.environ.get("GATHER_CONFIG", DEFAULT_CONFIG)

def run_step(mod: str, tag: str, extra_env: dict | None = None):
    if not mod:
        return
    print(f"[RUN] {tag}: {mod} (subprocess)")
    env = os.environ.copy()
    env["GATHER_CONFIG"] = CONFIG_PATH

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"

    if extra_env:
        # prepend our values so they take priority
        for k, v in extra_env.items():
            if not v:
                continue
            env[k] = f"{v}{os.pathsep}{env[k]}" if env.get(k) else v

    subprocess.run([sys.executable, "-m", mod], check=True, cwd=str(REPO_ROOT), env=env)

def _resolve_from_config(p: str) -> str:
    cfg_dir = Path(CONFIG_PATH).resolve().parent
    pp = Path(p)
    return str(pp if pp.is_absolute() else (cfg_dir / pp).resolve())

def _openpose_env(openpose_root: Path) -> dict:
    py_dir = openpose_root / "build" / "python"
    return {"PYTHONPATH": str(py_dir)} if py_dir.exists() else {}

def main():
    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        C = json.load(f)

    src_key = "video_path" if C.get("video_path") else "frames_dir"
    print(f"[PRECOMPUTE] Using {src_key}: {C.get(src_key)}")

    steps = [
        ("ball",    "Ball"),
        ("hoop",    "Hoop"),
        ("court",   "Court"),
        ("scoring", "Scoring"),
        ("pose",    "Pose"),
    ]
    for key, tag in steps:
        spec = C.get(key, {}).get("module")
        if not spec:
            continue

        extra_env = None
        if key == "pose":
            op_root = Path(_resolve_from_config(C["pose"]["openpose_root"]))
            extra_env = _openpose_env(op_root)

        run_step(spec, tag, extra_env=extra_env)

if __name__ == "__main__":
    main()

