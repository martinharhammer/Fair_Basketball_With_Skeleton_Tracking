import json
import math
import os
from statistics import median
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# =========================
# CONFIG
# =========================
JSON_FILE   = "../precompute/output/score_input.json"
CLAMP_MIN   = 160
CLAMP_MAX   = 225
MAX_BOOST   = 0.50
NEUTRAL_ON_MISSING = 1.0
ROUND_FACTOR = 2
OUTPUT_TXT  = "../precompute/output/final_score.txt"


# =========================
# UTIL: formatting helpers
# =========================
def clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))

def fmt(v: Any, ndigits: Optional[int] = None) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}" if ndigits is not None else f"{v}"
    return str(v)

def line(w: int = 80, ch: str = "=") -> str:
    return ch * w

def make_table(rows: List[List[str]], headers: List[str]) -> str:
    # compute column widths
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(str(r[i]).ljust(widths[i]) for i in range(cols))

    out = []
    out.append(fmt_row(headers))
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)

def winner_from_totals(team_totals: Dict[str, float]) -> Optional[Tuple[str, float]]:
    if not team_totals:
        return None
    winner = max(team_totals.items(), key=lambda kv: kv[1])
    return winner


# =========================
# CORE LOGIC
# =========================
def load_events(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])

def extract_height_cm(ev: Dict[str, Any]) -> Optional[float]:
    hblock = ev.get("height_m")
    if isinstance(hblock, dict):
        # prefer integer cm if present
        if "est_height_cm" in hblock and hblock["est_height_cm"] is not None:
            return float(hblock["est_height_cm"])
        # else derive from meters
        if "est_height_m" in hblock and hblock["est_height_m"] is not None:
            try:
                return float(hblock["est_height_m"]) * 100.0
            except Exception:
                return None
    return None

def compute_median_center(heights_clamped: List[Optional[float]]) -> float:
    vals = [h for h in heights_clamped if h is not None]
    if not vals:
        # If no heights, center at the middle of clamp range
        return (CLAMP_MIN + CLAMP_MAX) / 2.0
    return float(median(vals))

def linear_easiness(h: Optional[float], center: float) -> float:
    """
    Map height to an easiness factor:
    - At center: 1.0
    - At clamp min: 1.0 + MAX_BOOST
    - At clamp max: 1.0 - MAX_BOOST
    Missing height -> NEUTRAL_ON_MISSING
    """
    if h is None:
        return NEUTRAL_ON_MISSING
    h = float(h)
    if h <= center:
        # shorter => factor in [1, 1+MAX_BOOST]
        denom = max(center - CLAMP_MIN, 1e-9)
        frac = (center - h) / denom
        return 1.0 + MAX_BOOST * frac
    else:
        # taller => factor in [1-MAX_BOOST, 1]
        denom = max(CLAMP_MAX - center, 1e-9)
        frac = (h - center) / denom
        return 1.0 - MAX_BOOST * frac

def round2(x: float) -> float:
    return float(f"{x:.{ROUND_FACTOR}f}")

def render_report(center: float,
                  rows: List[List[str]],
                  team_rows: List[List[str]],
                  team_totals: Dict[str, float]) -> str:

    out = []
    out.append(line(70, "="))
    out.append("HEIGHT-ADJUSTED SCORING REPORT".center(70))
    out.append(line(70, "="))
    out.append("")
    out.append("CONFIG".center(70, "-"))
    out.append(f"Clamp range (cm): [{CLAMP_MIN}, {CLAMP_MAX}]")
    out.append(f"Max boost (±):    {int(MAX_BOOST*100)}%")
    out.append(f"Neutral (missing height) factor: {NEUTRAL_ON_MISSING}")
    out.append(line(70, "-"))
    out.append(f"Robust center (median height, cm): {fmt(center, 1)}")
    out.append("")
    out.append("PER-EVENT DETAILS".center(70, "-"))
    headers = ["ID", "Time", "Team", "Pts", "H(cm)", "H_clamp", "Factor", "Weighted Pts"]
    out.append(make_table(rows, headers))
    out.append("")
    out.append("TEAM TOTALS (WEIGHTED)".center(70, "-"))
    out.append(make_table(team_rows, ["Team", "Weighted Pts"]))
    out.append("")
    out.append(line(70, "="))
    out.append("FINAL SCORE".center(70))
    out.append(line(70, "="))

    width = 70
    for team, total in sorted(team_totals.items(), key=lambda kv: (-kv[1], kv[0])):
        left = f"{team}"
        right = f"{round2(total)}"
        dots = "." * max(2, width - len(left) - len(right) - 2)
        out.append(f"{left} {dots} {right}")

    win = winner_from_totals(team_totals)
    if win:
        out.append(line(70, "-"))
        out.append(f"Winner: {win[0]}  with {round2(win[1])} weighted points")
    out.append(line(70, "="))

    return "\n".join(out)


def main():
    events = load_events(JSON_FILE)

    processed = []
    for ev in events:
        pts  = ev.get("points", 0) or 0
        team = ev.get("team", "UNKNOWN") or "UNKNOWN"
        ts   = str(ev.get("timestamp", "-")).split(".")[0]
        eid  = ev.get("event_id", None)

        h_cm_raw = extract_height_cm(ev)
        h_cm_clamped = clamp(h_cm_raw, CLAMP_MIN, CLAMP_MAX)

        processed.append({
            "event_id": eid,
            "time": ts,
            "team": team,
            "points": float(pts),
            "height_cm": h_cm_raw,
            "height_clamped": h_cm_clamped
        })

    center = compute_median_center([p["height_clamped"] for p in processed])

    team_totals = defaultdict(float)
    rows = []
    for p in processed:
        factor = linear_easiness(p["height_clamped"], center)
        wpts = p["points"] * factor

        team_totals[p["team"]] += wpts

        rows.append([
            str(p["event_id"]),
            p["time"],
            p["team"],
            fmt(p["points"], ROUND_FACTOR),
            fmt(p["height_cm"], 0),
            fmt(p["height_clamped"], 0),
            fmt(round2(factor), ROUND_FACTOR),
            fmt(round2(wpts), ROUND_FACTOR),
        ])

    team_rows = []
    for team, total in sorted(team_totals.items(), key=lambda kv: (-kv[1], kv[0])):
        team_rows.append([team, fmt(round2(total), ROUND_FACTOR)])

    report_text = render_report(center, rows, team_rows, team_totals)

    print(report_text)

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved scoring sheet to: {os.path.abspath(OUTPUT_TXT)}")


if __name__ == "__main__":
    main()

