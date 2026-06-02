import torch
import os
import numpy as np

from ultralytics import YOLO
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

print("\n===== GENERATING FGSM DATASET =====\n")

epsilon = 0.07

os.makedirs(
    "adversarial_images",
    exist_ok=True
)

yolo_model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

model = yolo_model.model
model.eval()

transform = transforms.ToTensor()

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

criterion = torch.nn.CrossEntropyLoss()

for idx in range(len(dataset)):

    image, label = dataset[idx]

    image = image.unsqueeze(0)

    image.requires_grad = True

    outputs = model(image)

    logits = outputs[0]

    target = torch.tensor([label])

    loss = criterion(
        logits,
        target
    )

    model.zero_grad()

    loss.backward()

    grad = image.grad.data

    adv = image + epsilon * grad.sign()

    adv = torch.clamp(
        adv,
        0,
        1
    )

    img = adv.squeeze().detach().cpu().numpy()

    img = np.transpose(
        img,
        (1, 2, 0)
    )

    img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(
        f"adversarial_images/{idx}.png"
    )

    if idx % 1000 == 0:
        print(f"Processed {idx}")

print("\nFGSM dataset generated successfully.")