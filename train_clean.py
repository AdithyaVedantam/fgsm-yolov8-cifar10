from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="cifar10",
    epochs=20,
    imgsz=64,
    batch=64,
    device="mps",
    name="clean_baseline"
)

print("Clean model training complete")