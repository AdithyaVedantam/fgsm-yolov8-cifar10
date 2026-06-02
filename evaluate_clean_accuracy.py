from ultralytics import YOLO

print("\n===== CLEAN MODEL EVALUATION =====\n")

model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

metrics = model.val(data="cifar10")

accuracy = metrics.top1 * 100

print(f"Clean Accuracy: {accuracy:.2f}%")