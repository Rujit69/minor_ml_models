#model dynamically updates graph and saves best model based on accuracy

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for headless environments
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# Section 1: Data Loading and Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'D:\hey vagwan jei sri ganesh\images'  # Path to the dataset
train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class_names = train_dataset.classes

# Section 2: Define Model
model = models.densenet121(weights='IMAGENET1K_V1')
for param in model.features[:-5].parameters():  # Unfreeze only the last 5 layers
    param.requires_grad = True
model.classifier = nn.Linear(1024, 2)  # Binary classifier for real/fake
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Section 3: Define Loss, Optimizer, and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Section 4: Training Function with Dynamic Graph Updates and Best Model Saving
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, checkpoint_path='model4.pth', history_path='training_history.pth', best_model_path='best_model.pth'):
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    best_accuracy = 0  # Track the best accuracy
    scaler = GradScaler()  # For mixed precision training

    # Load training history if it exists
    if os.path.exists(history_path):
        print(f"Loading training history from {history_path}...")
        history = torch.load(history_path)
        train_losses = history['train_losses']
        train_accuracies = history['train_accuracies']
        start_epoch = history.get('epoch', 0)
        best_accuracy = history.get('best_accuracy', 0)
        print(f"Resuming training from epoch {start_epoch + 1} with best accuracy {best_accuracy:.2f}%...")
    else:
        print("No training history found, starting from scratch.")

    # Check if a checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch + 1}...")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training Loop
    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch  # Track the actual epoch number
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {current_epoch + 1}/{start_epoch + num_epochs}")

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale loss and backpropagate
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scale for next iteration

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
        train_losses.append(epoch_loss)  # Append loss for this epoch
        train_accuracies.append(epoch_acc)  # Append accuracy for this epoch

        print(f"Epoch {current_epoch + 1}/{start_epoch + num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        scheduler.step()

        # Save the model checkpoint after every epoch
        torch.save({
            'epoch': current_epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {current_epoch + 1}")

        # Save the best model based on accuracy
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save({
                'epoch': current_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_accuracy': best_accuracy,
            }, best_model_path)
            print(f"New best model saved with accuracy {best_accuracy:.2f}%")

        # Save the training history after every epoch
        torch.save({
            'epoch': current_epoch + 1,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'best_accuracy': best_accuracy,
        }, history_path)
        print(f"Training history saved at epoch {current_epoch + 1}")

        # Plot the training loss and accuracy
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, current_epoch + 2), train_losses, label='Training Loss')  # Use current_epoch + 1 for correct numbering
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, current_epoch + 2), train_accuracies, label='Training Accuracy')  # Use current_epoch + 1 for correct numbering
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_plot.png')  # Save the plot to a file
        print("Plot saved to 'training_plot.png'")

# Section 6: Train the Model
if __name__ == '__main__':
    checkpoint_path = 'model4.pth'
    history_path = 'training_history.pth'
    best_model_path = 'best_model.pth'
    num_epochs = 1  # Run 3 epochs today
    train_model(model, train_loader, criterion, optimizer, device, num_epochs, checkpoint_path, history_path, best_model_path)