from ultralytics import YOLO
import cv2, glob, os, sys

# paths
model_path   = "../runs/detect/train6/weights/best.pt"
input_folder = "/home/ubuntu/ball_tracking/videos/china_indonesia/frames"
output_folder = "../output"
os.makedirs(output_folder, exist_ok=True)

# gather images
patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
img_paths = sorted([p for pat in patterns for p in glob.glob(os.path.join(input_folder, pat))])
print(f"Found {len(img_paths)} images")
if not img_paths:
    sys.exit("No images found.")

# load model
model = YOLO(model_path)

# inference + save annotated
for img_path in img_paths:
    result = model(img_path, imgsz=960, conf=0.10, iou=0.60, verbose=False)[0]
    annotated_bgr = result.plot()  # already BGR
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    ok = cv2.imwrite(save_path, annotated_bgr)
    print(("Saved " if ok else "Failed to save ") + save_path)

