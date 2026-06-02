from ultralytics import YOLO
from torchvision.datasets import CIFAR10
from torchvision import transforms

print("\n===== FGSM ATTACK TEST =====\n")

model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

transform = transforms.ToTensor()

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, label = testset[0]

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print(f"True Label: {class_names[label]}")

# Clean prediction
results = model.predict(
    source="adversarial_images/fgsm_example.png",
    verbose=False
)

prediction = results[0].probs.top1

print(f"FGSM Prediction: {class_names[prediction]}")
print(f"Confidence: {results[0].probs.top1conf:.4f}")