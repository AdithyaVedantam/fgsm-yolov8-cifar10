import torch
from ultralytics import YOLO
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import numpy as np
import os

print("\n===== GENERATING FGSM ADVERSARIAL IMAGE =====\n")

# Load trained model
yolo_model = YOLO(
    "runs/classify/clean_baseline/weights/best.pt"
)

# Get underlying PyTorch model
model = yolo_model.model
model.eval()

# Load CIFAR10 test dataset
transform = transforms.ToTensor()

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Select first image
image, label = testset[0]

# Add batch dimension
image = image.unsqueeze(0)

# Enable gradients
image.requires_grad = True

# Forward pass
outputs = model(image)

# YOLO classification returns a tuple
logits = outputs[0]

# Compute loss
criterion = torch.nn.CrossEntropyLoss()

target = torch.tensor([label])

loss = criterion(logits, target)

# Backpropagation
model.zero_grad()

loss.backward()

# FGSM
epsilon = 0.03

data_grad = image.grad.data

perturbed_image = image + epsilon * data_grad.sign()

perturbed_image = torch.clamp(
    perturbed_image,
    0,
    1
)

# Create folder
os.makedirs(
    "adversarial_images",
    exist_ok=True
)

# Convert tensor to image
img = perturbed_image.squeeze().detach().cpu().numpy()

img = np.transpose(
    img,
    (1, 2, 0)
)

img = (img * 255).astype(np.uint8)

save_path = "adversarial_images/fgsm_example.png"

Image.fromarray(img).save(save_path)

print(f"Original Label: {label}")
print(f"Epsilon: {epsilon}")
print(f"Saved Image: {save_path}")