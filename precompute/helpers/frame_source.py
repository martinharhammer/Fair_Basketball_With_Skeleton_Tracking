import os, glob, cv2
from collections import deque

class FrameSource:
    """
    Unified reader for either a video file or a frames folder (PNG/JPG).
    Yields (idx, name, frame) where name looks like 'frame_000123.png'.
    """
    def __init__(self, video_path=None, frames_dir=None):
        print(f"video: {video_path}, frames: {frames_dir}")
        assert (video_path is None) ^ (frames_dir is None), "Provide either video_path or frames_dir (not both)"
        self.mode = "video" if video_path else "folder"
        self.video_path = video_path
        self.frames_dir = frames_dir
        if self.mode == "video":
            print("video")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        else:
            print("frames")
            exts = ("*.png","*.jpg","*.jpeg")
            paths = []
            for e in exts: paths += glob.glob(os.path.join(frames_dir, e))
            self.paths = sorted(paths)
            self.count = len(self.paths)
            self.fps = None

    def __iter__(self):
        if self.mode == "video":
            i = 0
            while True:
                ok, frame = self.cap.read()
                if not ok: break
                yield i, f"frame_{i:06d}.png", frame
                i += 1
            self.cap.release()
        else:
            print("frames")
            for i, p in enumerate(self.paths):
                img = cv2.imread(p)
                if img is None: continue
                yield i, os.path.basename(p), img

def read_frame_at(video_path: str, index: int):
    """Random-access read of 1 frame by absolute index (for pose backtrack)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, img = cap.read()
    cap.release()
    return img if ok else None

def sliding_windows(source_iter, window: int, step: int = 1):
    """
    Turn a stream into overlapping windows of length 'window'.
    Yields (center_idx, names_list, frames_list) with center at mid.
    """
    buf = deque()
    names = deque()
    mid = window // 2
    for idx, name, frame in source_iter:
        buf.append(frame); names.append(name)
        if len(buf) < window: continue
        center = idx - (window - 1 - mid)
        yield center, list(names), list(buf)
        for _ in range(step): 
            if buf: buf.popleft()
            if names: names.popleft()

