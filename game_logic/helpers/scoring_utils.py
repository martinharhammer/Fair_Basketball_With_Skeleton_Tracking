import cv2

def get_video_fps_strict(video_path: str) -> float:
    """
    Open video and read FPS.
    Fail fast if video cannot be opened or FPS is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[FPS] Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"[FPS] Invalid FPS reported for: {video_path}")
    return float(fps)

def hms_from_frame(idx: int, fps: float) -> str:
    """
    Convert frame index + fps into HH:MM:SS.mmm string.
    """
    t_s = idx / fps
    ms = int(round((t_s - int(t_s)) * 1000))
    s  = int(t_s) % 60
    m  = (int(t_s) // 60) % 60
    h  = int(t_s) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def points_to_value(label: str) -> int:
    """
    Map points label to numeric value.
    """
    return {"1PT": 1, "2PT": 2, "3PT": 3}.get(label, 0)

