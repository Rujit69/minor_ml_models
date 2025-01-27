#finally asked gpt to improve the code 
#dynamic plot works but needs more testing

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# Section 1: Data Loading and Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'D:/hey vagwan jei sri ganesh/images'  # Path to the dataset
train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
class_names = train_dataset.classes

# Section 2: Define Model
model = models.densenet121(weights='IMAGENET1K_V1')

# Unfreeze only the last few layers for fine-tuning
for param in model.features[:-5].parameters():
    param.requires_grad = False

model.classifier = nn.Linear(1024, 2)  # Binary classifier for real/fake
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Section 3: Define Loss, Optimizer, and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Section 4: Training Function with Dynamic Graph Updates and Best Model Saving
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, checkpoint_path='model4.pth', best_model_path='best_model.pth'):
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    best_accuracy = 0.0
    scaler = GradScaler()  # For mixed precision training

    # Check if a checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resuming training from epoch {start_epoch + 1}...")

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Backpropagate
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update scaling factor

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Accuracy': f"{(correct / total * 100):.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        scheduler.step()

        # Save the best model based on accuracy
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_accuracy': best_accuracy,
            }, best_model_path)
            print(f"New best model saved with accuracy {best_accuracy:.2f}%")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'best_accuracy': best_accuracy,
        }, checkpoint_path)

    # Plot the training loss and accuracy at the end of all epochs
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png')  # Save the plot to a file
    print("Plot saved to 'training_plot.png'")

# Section 5: Train the Model
if __name__ == '__main__':
    checkpoint_path = 'model4.pth'
    best_model_path = 'best_model.pth'
    num_epochs = 3
    train_model(model, train_loader, criterion, optimizer, device, num_epochs, checkpoint_path, best_model_path)
