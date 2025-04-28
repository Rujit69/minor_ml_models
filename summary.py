# just to verify the number of features extracted by dense net 121

import torch
import torchvision.models as models
import torch.nn.functional as F

# Load DenseNet-121 with pretrained ImageNet weights
model = models.densenet121(weights=True)

# Use only the feature extractor part of DenseNet-121
feature_extractor = model.features
feature_extractor.eval()  # Set model to evaluation mode

# Create a dummy input image (batch_size=1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Forward pass through the feature extractor
with torch.no_grad():
    feature_map = feature_extractor(dummy_input)

# Print the shape of the feature map
print("Feature map shape before GAP:", feature_map.shape)


# Apply global average pooling manually
gap = F.adaptive_avg_pool2d(feature_map, (1, 1))
gap_vector = gap.view(gap.size(0), -1)


print("Feature vector shape after GAP:", gap_vector.shape)

