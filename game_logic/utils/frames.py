import re

_FRAME_RE = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)

def frame_name_to_index(name: str) -> int:
    m = _FRAME_RE.search(name)
    if not m:
        raise ValueError(f"Bad frame name: {name}")
    return int(m.group(1))

def hms_from_frame(idx: int, fps: float) -> str:
    t_s = idx / fps
    ms = int(round((t_s - int(t_s)) * 1000))
    s  = int(t_s) % 60
    m  = (int(t_s) // 60) % 60
    h  = int(t_s) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
