import torch
from torchvision import transforms, models
from PIL import Image

# Define the necessary transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of DenseNet
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Path to the saved model checkpoint
checkpoint_path = 'best_model.pth'

# Load the model once and reuse it
def load_model():
    # Load the pre-trained DenseNet model
    model = models.densenet121(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers except the final classifier

    # Modify the classifier layer for binary classification (real vs fake)
    model.classifier = torch.nn.Linear(1024, 2)  # Two output classes (real and fake)

    # Load the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    return model, device

# Load the model once and store it in a global variable
model, device = load_model()

# List of class names based on folder structure
class_names = ['fake', 'real']

# Function to predict on a new image
def predict_image(image_path):
    try:
        # Load the image
        image = Image.open(image_path)

        # Convert to RGB if the image has an alpha channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = data_transforms(image)  # Apply the transformations to the image
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)  # Move to the appropriate device (GPU or CPU)

        # Predict
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class index
            return class_names[predicted]
    except Exception as e:
        return f"Error during prediction: {e}"