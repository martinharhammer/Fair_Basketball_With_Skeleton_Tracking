from __future__ import annotations
import os, json
from typing import Iterable, List
from .frames import frame_name_to_index

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def append_jsonl(path: str, rows: list[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def collect_frames_from_jsonls(*paths: Iterable[str]) -> List[str]:
    names = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                nm = row.get("frame")
                if isinstance(nm, str):
                    names.add(nm)
                frs = row.get("frames")
                if isinstance(frs, list):
                    for fr in frs:
                        nm2 = fr.get("frame")
                        if isinstance(nm2, str):
                            names.add(nm2)
    return sorted(names, key=frame_name_to_index)

