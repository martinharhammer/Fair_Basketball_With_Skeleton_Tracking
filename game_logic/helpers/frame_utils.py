# game_logic/helpers/frame_utils.py
import re

_FRAME_RE = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)

def frame_name_to_index(name: str) -> int:
    m = _FRAME_RE.search(name)
    if not m:
        raise ValueError(f"Bad frame name: {name}")
    return int(m.group(1))

