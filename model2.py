#THIS FILE IS USED TO TRAINING THE ML MODEL

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Section 1: Data Loading and Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'D:\hey vagwan jei sri ganesh\images'  # Path to the 'image' folder containing 'fake' and 'real' subfolders
train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms) # automatically assigns fake=0, real=1
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
class_names = train_dataset.classes

# Section 2: Define Model
model = models.densenet121(weights='IMAGENET1K_V1')  # Updated to use 'weights' argument
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(1024, 2)  # Binary classifier for real/fake
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Section 3: Define Loss, Optimizer, and Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Section 4: Training Function with Checkpoint Saving
# Section 4: Training Function with Checkpoint Saving
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, checkpoint_path='fake_face_detector.pth'):
    start_epoch = 0  # Start from epoch 0 if no checkpoint is found

    # Check if a checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)

        # Load the model state dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error: 'model_state_dict' not found in checkpoint!")

        # Load the optimizer state dictionary
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Error: 'optimizer_state_dict' not found in checkpoint!")

        # Load the epoch number
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(dataloader)

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == num_batches:  # Print every 10 batches or at the end
                print(f"Processed {batch_idx + 1}/{num_batches} batches ({(batch_idx + 1) * dataloader.batch_size}/{len(dataloader.dataset)} images)")

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save the model after every epoch
        torch.save({
            'epoch': epoch + 1,  # Save the current epoch (use epoch+1 to match the display)
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# Section 6: Train and Test the Model
if __name__ == '__main__':
    num_epochs = 27
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
