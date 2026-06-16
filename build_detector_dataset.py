import os
import shutil
from torchvision.datasets import CIFAR10

print("\n===== BUILDING DETECTOR DATASET =====\n")

# Create folders

folders = [
    "detector_dataset/train/clean",
    "detector_dataset/train/adversarial",
    "detector_dataset/val/clean",
    "detector_dataset/val/adversarial"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Load CIFAR10

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True
)

# First 8000 -> train
# Last 2000 -> validation

for idx in range(10000):

    clean_img, _ = dataset[idx]

    adv_path = f"adversarial_images/{idx}.png"

    if idx < 8000:

        clean_save = (
            f"detector_dataset/train/clean/{idx}.png"
        )

        adv_save = (
            f"detector_dataset/train/adversarial/{idx}.png"
        )

    else:

        clean_save = (
            f"detector_dataset/val/clean/{idx}.png"
        )

        adv_save = (
            f"detector_dataset/val/adversarial/{idx}.png"
        )

    clean_img.save(clean_save)

    shutil.copy(
        adv_path,
        adv_save
    )

    if idx % 1000 == 0:
        print(f"Processed {idx}")

print("\nDetector dataset built successfully.")