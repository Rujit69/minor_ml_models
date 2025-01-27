import os
import torch
from torchvision import models, transforms
from PIL import Image

# Image transformations for validation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the pre-trained model
model = models.densenet121(weights='IMAGENET1K_V1')
model.classifier = torch.nn.Linear(1024, 2)  # Binary classification (real vs fake)

# Load trained weights
def load_model(checkpoint_path='fake_face_detector.pth'):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("Checkpoint not found. Ensure model is trained and saved.")
        exit()

load_model()

# Function for image prediction
def predict_image(image_path, model, class_names):
    model.eval()

    if os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        print(f"File not found: {image_path}")
        return None

    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    outputs = model(image)
    print(f"Raw outputs: {outputs}")  # Print the raw output

    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

    print(f"Predicted class: {predicted_class}")
    return predicted_class

# Validation function
def validate_on_folder(folder_path, class_names):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):
            print(f"Processing image: {image_path}")
            predicted_class = predict_image(image_path, model, class_names)
            if predicted_class is not None:
                true_class = image_path.split(os.path.sep)[-2]  # Assuming the folder name is the class label (fake or real)
                print(f"True class: {true_class}")
                if predicted_class == true_class:
                    correct += 1
                total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Main function for validation
if __name__ == '__main__':
    folder_path = 'D:\hey vagwan jei sri ganesh\test_images'  # Path to validation images
    class_names = ['fake', 'real']  # Class names for prediction, update as necessary

    # Validate the model on the given folder
    validate_on_folder(folder_path, class_names)
