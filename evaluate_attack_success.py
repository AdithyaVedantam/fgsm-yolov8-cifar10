from ultralytics import YOLO
from torchvision.datasets import CIFAR10

print("\n===== FGSM ATTACK SUCCESS RATE =====\n")

model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True
)

successful_attacks = 0
total = 0

for idx in range(len(dataset)):

    _, label = dataset[idx]

    image_path = f"adversarial_images/{idx}.png"

    results = model.predict(
        source=image_path,
        verbose=False
    )

    prediction = results[0].probs.top1

    if prediction != label:
        successful_attacks += 1

    total += 1

    if idx % 1000 == 0:
        print(f"Processed {idx}")

asr = (successful_attacks / total) * 100

print(f"\nAttack Success Rate (ASR): {asr:.2f}%")