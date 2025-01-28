import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

def load_validation_data(val_dir, batch_size=64):
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load validation data
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return val_loader, val_dataset.classes

def validate(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        progress_bar = tqdm(val_loader, desc="Validating", unit="batch")  # Progress bar
        for val_images, val_labels in progress_bar:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels)
            total_loss += loss.item() * val_images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == val_labels).sum().item()
            total += val_labels.size(0)

            # Update progress bar with loss and accuracy
            progress_bar.set_postfix(loss=loss.item(), accuracy=(correct / total * 100) if total > 0 else 0)

    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total * 100 if total > 0 else 0
    return avg_loss, accuracy

if __name__ == "__main__":
    val_dir = "D:/hey vagwan jei sri ganesh/validation images"  # Update this to your validation directory
    batch_size = 64  # Adjust as needed

    # Load validation data
    val_loader, class_names = load_validation_data(val_dir, batch_size)

    # Load your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(1024, 2)  # Ensure the model matches your training setup
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])  # Load best model
    model.to(device)

    # Validate the model
    val_loss, val_accuracy = validate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")