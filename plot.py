
#! nothing fancy here just a code to plot and view data saved in training_history.pth file
#! ps already training garney code mai plot ko codes cha hai this for extra details saved in 
#! training_history.pth

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

history_path = 'training_history.pth'
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

# Plot the training and validation loss and accuracy at the end of all epochs
plt.figure(figsize=(12, 5))
# Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xticks(range(1, len(train_losses) + 1))  # Set x-axis ticks starting from 1
plt.grid(True)
plt.legend()

# Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='blue', marker='o')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xticks(range(1, len(train_accuracies) + 1))  # Set x-axis ticks starting from 1
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('detailed_plot.png')  # Save the plot to a file
print("Plot saved to 'detailed_plot.png'")

#! plot garda use huney data display garna ko lagi
file_path = 'training_history.pth'
saved_data = torch.load(file_path)
print(saved_data)






