from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64


app = Flask(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetClassifier, self).__init__()
        # Load pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Modify the classifier to match the number of output classes
        self.efficientnet._fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Forward pass through EfficientNet
        out = self.efficientnet(x)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetClassifier, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the classifier to match the number of output classes
        self.resnet.fc = nn.Linear(512, num_classes)  # Modify the fully connected layer

    def forward(self, x):
        # Forward pass through ResNet
        out = self.resnet(x)
        return out

class FeatureFusion(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeatureFusion, self).__init__()
        # Fusion layer
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.fc(x)
        return out

class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes=7, fusion_out_features=256):
        super(EnsembleClassifier, self).__init__()
        self.efficientnet_classifier = EfficientNetClassifier(num_classes)
        self.resnet_classifier = ResNetClassifier(num_classes)
        
        # Fusion layer
        self.fusion_layer = FeatureFusion(in_features=1000 + 7, out_features=fusion_out_features)

        # Final classifier
        self.fc = nn.Linear(fusion_out_features, num_classes)

    def forward(self, x):
        # Forward pass through both classifiers
        efficientnet_out = self.efficientnet_classifier(x)
        resnet_out = self.resnet_classifier(x)
        
        # Concatenate features
        combined_features = torch.cat((efficientnet_out, resnet_out), dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        # Final classification
        out = self.fc(fused_features)
        return out

# Example usage
ensemble_classifier = EnsembleClassifier()


# Define class names
class_names ={0: "Actinic_keratosis",
    1: "Basal_cell_carcinoma",
    2: "Benign_keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Nervus",
    6: "Vascular_lesion" }


ensemble_classifier = EnsembleClassifier()
ensemble_classifier.load_state_dict(torch.load('best_model_epoch_2[efficientnet+resnet].pt', map_location=torch.device('cpu')))


# Image preprocessing
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Read image file
        img_bytes = file.read()
        
        # Convert image bytes to PIL image
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess the image
        input_image = test_transform(pil_image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            output = ensemble_classifier(input_image)

        # Get predicted class and confidence
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = F.softmax(output, dim=1)[0][predicted_class].item()
        predicted_class_name = class_names[predicted_class]
        
        
        
        # Convert PIL image to base64 for display
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return render_template('result.html', original_image=img_str, predicted_class=predicted_class_name, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
