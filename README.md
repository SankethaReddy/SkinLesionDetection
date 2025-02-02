# Skin Lesion Recognition using EfficientNet + ResNet

## Overview
Skin lesions and skin cancer pose significant health risks, requiring accurate and timely detection for effective treatment. This project presents a deep learning-based skin lesion classification system that leverages EfficientNet and ResNet architectures to enhance diagnostic accuracy. The proposed model is designed to classify benign and malignant skin lesions using dermoscopic images from the ISIC 2018 challenge dataset.

## Features
- **Ensemble Model:** Combines EfficientNet and ResNet architectures to enhance feature extraction and classification accuracy.
- **Data Augmentation:** Includes resizing, normalization, flipping, rotation, and color jittering to improve model generalization.
- **Feature Fusion Layer:** Integrates features from both architectures to create a robust classification system.
- **Optimized for Deployment:** Suitable for real-time applications in telemedicine and mobile health monitoring.

## Dataset
The model is trained using the ISIC 2018 challenge dataset, which includes a diverse set of benign and malignant skin lesion images. The dataset undergoes preprocessing to handle class imbalance and improve model performance.
Dataset link:https://challenge.isic-archive.com/data/#2018

## Model Architecture
1. **EfficientNet:** Efficiently scales network depth, width, and resolution to achieve high accuracy with fewer parameters.
2. **ResNet:** Employs residual connections to facilitate training deeper networks without degradation.
3. **Fusion Layer:** Merges feature maps from EfficientNet and ResNet to enhance classification accuracy.
4. **Final Classifier:** A fully connected layer that outputs the final prediction.

## Training Process
- **Data Preprocessing:** Standardizes images and applies augmentation techniques.
- **Transfer Learning:** Fine-tunes pre-trained EfficientNet and ResNet models.
- **Hyperparameter Tuning:** Optimizes batch size, learning rate, and dropout rates.
- **Evaluation Metrics:** Uses accuracy, precision, recall, and F1-score to assess model performance.

## Results
| Model | Accuracy |
|--------|---------|
| EfficientNet | 82.86% |
| InceptionV4 | 67 |
| MobileNetV2 | 81.88% |
| MobileNetV2 + Unet | 71.94% |
| **EfficientNet + ResNet (Proposed)** | **95.89%** |

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/skin-lesion-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd skin-lesion-recognition
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the model:
   ```bash
   python run app.py
   ```

## Deployment
The model is deployed using Flask for real-time classification of skin lesion images. A user-friendly web interface allows users to upload images and receive predictions instantly.




