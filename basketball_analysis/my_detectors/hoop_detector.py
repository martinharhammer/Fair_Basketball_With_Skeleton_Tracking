# my_detectors/hoop_detector.py  (or trackers/hoop_detector.py)
from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append('../')  # keep if your utils isn’t a proper package yet
from utils import read_stub, save_stub

class HoopDetector:
    """
    YOLO-based hoop detector (batched, cached).
    Filters strictly to class name "Hoop" and writes per-frame detections to a stub.
    """

    def __init__(self, model_path=None, conf=0.10, iou=0.60, imgsz=960, batch_size=20, keep_top_k=1):
        # Use provided model path, else try configs, else fallback to a sensible default
        if model_path is None:
            try:
                from configs import HOOP_DETECTOR_PATH
                model_path = HOOP_DETECTOR_PATH
            except Exception:
                model_path = "../runs/detect/train/weights/best.pt"

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.keep_top_k = keep_top_k
        self.target_class_name = "hoop"

        self.model = YOLO(self.model_path)

    def _detect_in_batches(self, frames):
        preds = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            out = self.model.predict(
                batch,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False
            )
            preds.extend(out)
        return preds

    def get_detections(self, frames, read_from_stub=False, stub_path=None):
        """
        Returns per-frame hoop detections and caches them.
        Output per frame:
            [
              {"bbox":[x1,y1,x2,y2], "conf": float},  # up to keep_top_k items
              ...
            ]
        """
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        preds = self._detect_in_batches(frames)
        results = []

        for det in preds:
            sv_det = sv.Detections.from_ultralytics(det)
            keep = []
            for i in range(len(sv_det)):
                class_id = int(sv_det.class_id[i])
                class_name = det.names[class_id]
                if class_name == self.target_class_name:
                    keep.append({
                        "bbox": sv_det.xyxy[i].tolist(),    # [x1, y1, x2, y2]
                        "conf": float(sv_det.confidence[i]),
                    })

            keep.sort(key=lambda r: r["conf"], reverse=True)
            results.append(keep[:self.keep_top_k] if self.keep_top_k else keep)

        save_stub(stub_path, results)
        return results

