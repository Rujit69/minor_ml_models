import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_validation_data(val_dir, batch_size=64):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return val_loader, val_dataset.classes

def validate(model, val_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")
        for val_images, val_labels in progress_bar:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels)
            total_loss += loss.item() * val_images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == val_labels).sum().item()
            total += val_labels.size(0)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item(), accuracy=(correct / total * 100) if total > 0 else 0)

    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total * 100 if total > 0 else 0

    return avg_loss, accuracy, all_labels, all_preds

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved as {filename}")

if __name__ == "__main__":
    val_dir = r"D:\hey vagwan jei sri ganesh\validation images"
    batch_size = 64

    val_loader, class_names = load_validation_data(val_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(512, 2)
)
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.to(device)

    val_loss, val_accuracy, all_labels, all_preds = validate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)