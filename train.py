import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor()
])

# Load Dataset (Replace with your dataset path)
train_dataset = ImageFolder(root="/home/jinwoo/Desktop/hand-sign-control-ML/train", transform=transform)
val_dataset = ImageFolder(root="/home/jinwoo/Desktop/hand-sign-control-ML/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define CNN Model
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust for image size (128x128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model
num_classes = len(train_dataset.classes)
model = GestureCNN(num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print Batch Progress
        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print Epoch Loss
    print(f"Epoch [{epoch+1}/{epochs}] Completed - Average Loss: {running_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "gesture_model.pth")
print("Model Saved Successfully!")
