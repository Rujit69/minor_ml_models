#reducing overfitting 
# Optimized for performance, shows graph
#! note to self to generate new plot for new dataset delete training_history file this wont effect the training process as checkpoint is loaded from model itself

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

# Import the validation function from the validation script
from validation import validate, load_validation_data   

# Section 1: Data Loading and Preprocessing

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation
    transforms.RandomCrop(224, padding=4),  # Random cropping with padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color changes
    transforms.RandomVerticalFlip(), 
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Random Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = r'D:\hey vagwan jei sri ganesh\Training Dataset' 
train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
class_names = train_dataset.classes

# Section 2: Load Validation Data
val_dir = r'D:\hey vagwan jei sri ganesh\validation images'  # Update to your validation directory
val_loader, _ = load_validation_data(val_dir)

# Section 3: Define Model
model = models.densenet121(weights='IMAGENET1K_V1')

# Unfreeze only the last few layers for fine-tuning
#for param in model.features[:-5].parameters():
   # param.requires_grad = False

    # Freeze most of the layers

    #! for better results 
for param in model.features[:-5].parameters():
    param.requires_grad = False  # Freeze early layers

# Unfreeze the last 5 layers in the convolutional part and the classifier
for param in model.features[-5:].parameters():
    param.requires_grad = True  # Unfreeze last 5 convolutional layers

for param in model.classifier.parameters():
    param.requires_grad = True  # Unfreeze classifier layers


#model.classifier = nn.Linear(1024, 2)  # Binary classifier for real/fake

#!to reduce overfitting
model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(512, 2)
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Section 4: Define Loss, Optimizer, and Learning Rate Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#! to reduce overfitting
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


# Section 5: Training Function with Dynamic Graph Updates and Best Model Saving
def train_model(model, dataloader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_path='model8.pth', best_model_path='best_model.pth', history_path='training_history.pth'):
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    scaler = GradScaler()  # For mixed precision training

    # Load training history if it exists
    if os.path.exists(history_path):
        print(f"Loading training history from {history_path}...")
        history = torch.load(history_path)
        train_losses = history['train_losses']
        train_accuracies = history['train_accuracies']
        val_losses = history.get('val_losses', [])
        val_accuracies = history.get('val_accuracies', [])
        start_epoch = history.get('epoch', 0)
        best_accuracy = history.get('best_accuracy', 0.0)
        print(f"Resuming plotting from epoch {start_epoch + 1} with best accuracy {best_accuracy:.2f}%...")
    else:
        print("No training history found, creating training history from scratch.")

    # Check if a checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch + 1}...")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training Loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")

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

        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_loss)

        # Save the best model based on accuracy
        if val_accuracy  > best_accuracy:
            best_accuracy = val_accuracy 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'lr_scheduler_state_dict':scheduler.state_dict(),
                'best_accuracy': best_accuracy,
            }, best_model_path)
            print(f"New best model saved with validation accuracy {best_accuracy:.2f}%")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'lr_scheduler_state_dict':scheduler.state_dict(),
            'best_accuracy': best_accuracy,
        }, checkpoint_path)

        # Save training history
        torch.save({
            'epoch': epoch + 1,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_accuracy,
        }, history_path)
        print(f"Training history saved at epoch {epoch + 1}")


    # Plot the training and validation loss and accuracy at the end of all epochs
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss 
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses) + 1), [0] + train_losses, label='Training Loss', color='blue')  # Training loss in blue
    plt.plot(range(len(val_losses) + 1), [0] + val_losses, label='Validation Loss', color='orange')  # Validation loss in orange
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies) + 1), [0] + train_accuracies, label='Training Accuracy', color='blue')  # Training accuracy in blue
    plt.plot(range(len(val_accuracies) + 1), [0] + val_accuracies, label='Validation Accuracy', color='orange')  # Validation accuracy in orange
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_plot.png')  # Save the plot to a file
    print("Plot saved to 'training_validation_plot.png'")

# Section 6: Train the Model
if __name__ == '__main__':
    checkpoint_path = 'model8.pth'
    best_model_path = 'best_model.pth'
    history_path = 'training_history.pth'
    num_epochs = 1  # Number of epochs to run
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path, best_model_path, history_path)
