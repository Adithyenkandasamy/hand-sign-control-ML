import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
])

# Load dataset
train_dataset = ImageFolder(root="/home/jinwoo/Desktop/hand-sign-control-ML/train", transform=transform)
val_dataset = ImageFolder(root="/home/jinwoo/Desktop/hand-sign-control-ML/val", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print class names
print("Classes:", train_dataset.classes)
