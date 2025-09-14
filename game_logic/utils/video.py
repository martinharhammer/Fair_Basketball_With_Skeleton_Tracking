import cv2
import os

def get_video_fps_strict(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[FPS] Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"[FPS] Invalid FPS reported for: {video_path}")
    return float(fps)

def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames, output_video_path: str, fps: float = 24.0):
    if not output_frames:
        return
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    h, w = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    for frame in output_frames:
        out.write(frame)
    out.release()

