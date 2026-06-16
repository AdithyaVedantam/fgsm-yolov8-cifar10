from ultralytics import YOLO

print("\n===== TRAINING FGSM DETECTOR =====\n")

model = YOLO("yolov8n-cls.pt")

model.train(
    data="detector_dataset",
    epochs=10,
    imgsz=32,
    batch=64,
    device="mps",
    name="fgsm_detector"
)

print("\nFGSM detector training complete.")