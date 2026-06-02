from ultralytics import YOLO
from torchvision.datasets import CIFAR10

print("\n===== FGSM ACCURACY EVALUATION =====\n")

model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True
)

correct = 0
total = len(dataset)

for idx in range(total):

    _, label = dataset[idx]

    image_path = f"adversarial_images/{idx}.png"

    results = model.predict(
        source=image_path,
        verbose=False
    )

    prediction = results[0].probs.top1

    if prediction == label:
        correct += 1

    if idx % 1000 == 0:
        print(f"Processed {idx}")

accuracy = (correct / total) * 100

print(f"\nFGSM Accuracy: {accuracy:.2f}%")