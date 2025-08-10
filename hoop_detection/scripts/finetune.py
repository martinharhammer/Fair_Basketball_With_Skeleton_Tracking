from ultralytics import YOLO

model = YOLO("yolov12n.pt")

results = model.train(
    data="/home/ubuntu/hoop_detection/datasets/hoop_dataset4/data.yaml",
    epochs=100
)
