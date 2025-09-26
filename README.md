# Hanacaraka Aksara Jawa Handwriting Detection

A deep learning project for recognizing handwritten Javanese script (Aksara Jawa/Hanacaraka) characters using a fine-tuned ResNet50V2 model, complete with an interactive web application for real-time predictions.

## Overview

This project implements a convolutional neural network to classify 20 different Javanese script characters. The model uses transfer learning with ResNet50V2 as the base architecture and achieves high accuracy (>96%) through data augmentation and fine-tuning techniques. An interactive Streamlit web application is included for easy testing and demonstration.

## Features

- **Deep Learning Model**: Fine-tuned ResNet50V2 for 20-class Javanese character classification
- **Interactive Web App**: Streamlit-based interface for real-time image classification
- **Comprehensive Training Pipeline**: Complete notebook with data preprocessing, training, and evaluation
- **Model Persistence**: Saved model and class mappings for deployment

## Dataset

The training data comes from two Kaggle datasets:
- [Javanese Script Aksara Jawa Augmented](https://www.kaggle.com/datasets/hannanhunafa/javanese-script-aksara-jawa-augmented)
- [Hanacaraka](https://www.kaggle.com/datasets/vzrenggamani/hanacaraka)

The dataset contains images of 20 different Javanese characters with the following split:
- **Training**: 70%
- **Validation**: 15%
- **Testing**: 15%

## Model Architecture

- **Base Model**: ResNet50V2 (pre-trained on ImageNet)
- **Input Shape**: 224 × 224 × 3
- **Architecture**:
  - ResNet50V2 base (frozen initially)
  - Global Average Pooling
  - Dense layer (256 units, ReLU activation, L2 regularization)
  - Dropout (0.5)
  - Output layer (20 units, Softmax activation)

## Training Strategy

### Phase 1: Feature Extraction
- Freeze ResNet50V2 base layers
- Train only the classifier head
- Learning rate: 1e-4

### Phase 2: Fine-tuning
- Unfreeze last 40 layers of ResNet50V2
- Fine-tune with lower learning rate: 1e-5
- Data augmentation applied during training

### Data Augmentation
- Rotation: ±20 degrees
- Zoom: 0.9-1.0
- Width/Height shift: ±20%
- Shear: ±20%
- ResNet50V2 preprocessing

### Callbacks
- **Early Stopping**: Monitor validation loss, patience=5
- **Learning Rate Reduction**: Factor=0.2, patience=3

## Requirements

```python
# Core dependencies
tensorflow>=2.x
opencv-python==4.11.0.86
pandas>=2.3.2
numpy
matplotlib
seaborn>=0.13.2
scikit-learn>=1.7.2

# For Streamlit app
streamlit>=1.50.0
pillow

# Development tools
git-filter-repo>=2.47.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Exilitys/Hanacaraka-Aksara-Jawa-Handwriting-Detection.git
cd Hanacaraka-Aksara-Jawa-Handwriting-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the dataset (for training):
   - Download the datasets from Kaggle
   - Place the data in `./hanacaraka/` directory

## Usage

### 🚀 Quick Start - Web Application

Run the interactive Streamlit app for instant character recognition:

```bash
streamlit run main.py
```

This will launch a web interface where you can:
- Upload images of Javanese characters (JPG, JPEG, PNG)
- Get real-time predictions with confidence scores
- View the uploaded image alongside the prediction

### 📚 Training the Model

Run the Jupyter notebook for complete training pipeline:
```bash
jupyter notebook "aksara-jawa-letter-classification (1).ipynb"
```

The notebook includes:
- Data loading and preprocessing
- Model architecture setup
- Training with transfer learning
- Comprehensive evaluation and visualization

### 🔧 Using Pre-trained Model

The repository includes pre-trained model files:
- `model/ResNet50-FineTune-40Fl.keras` - Trained model (https://drive.google.com/drive/folders/1bdIbwjfyi5Spclair9FxbQG3GU5toH93?usp=sharing)
- `model/class-indicies.pkl` - Class label mappings

## Web Application Features

### Interface
- **Simple Upload**: Drag and drop or browse for image files
- **Real-time Preview**: Immediate display of uploaded images
- **Instant Prediction**: Fast classification using the trained model
- **Clean Results**: Clear display of predicted character labels

### Supported Formats
- JPG/JPEG
- PNG
- Automatic resizing to 224×224 pixels
- ResNet50V2 preprocessing applied automatically

## Model Evaluation

The notebook provides comprehensive evaluation including:
- **Accuracy metrics** on test set
- **Classification reports** with precision, recall, F1-scores
- **Confusion matrix visualization**
- **ROC curves** for all 20 classes
- **Sample predictions** with true vs predicted labels

## Results

The model demonstrates strong performance on the test set with detailed classification reports and visualizations available in the notebook.

### Key Performance Features:
- Multi-class classification of 20 Javanese characters
- Transfer learning with ResNet50V2
- Comprehensive data augmentation
- Fine-tuning strategy for improved performance
- Real-time inference capability

## File Structure

```
├── aksara-jawa-letter-classification (1).ipynb  # Main training notebook
├── main.py                                      # Streamlit web application
├── model/                                       # Model directory
│   ├── ResNet50-FineTune-40Fl.keras           # Trained model
│   └── class-indicies.pkl                      # Class mappings
├── hanacaraka/                                  # Dataset directory (for training)
├── .python-version                             # Python version
├── pyproject.toml                              # Project configuration
├── uv.lock                                     # Dependency lock file
└── README.md                                   # This file
```

## Screenshots

### Web Application Interface
The Streamlit app provides an intuitive interface for testing the model with your own images.

### Model Training Results
The notebook includes detailed visualizations of training progress, confusion matrices, and ROC curves.

## API Usage

You can also use the model programmatically:

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import pickle

# Load model and class mappings
model = load_model('model/ResNet50-FineTune-40Fl.keras')

with open('model/class-indicies.pkl', 'rb') as f:
    class_indicies = pickle.load(f)

label_map = {v: k for k, v in class_indicies}

# Preprocess and predict
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get prediction
pred_probs = model.predict(img_array)[0]
pred_label = label_map[np.argmax(pred_probs)]
confidence = np.max(pred_probs)

print(f"Predicted: {pred_label} (Confidence: {confidence:.2f})")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle dataset contributors
- TensorFlow/Keras community
- ResNet50V2 architecture by Microsoft Research
- Streamlit for the amazing web app framework

## Future Improvements

- [ ] Add confidence score display in web app
- [ ] Implement batch processing for multiple images
- [ ] Add character segmentation for full text recognition
- [ ] Deploy to cloud platforms (Heroku, Streamlit Cloud)
- [ ] Add data visualization dashboard
- [ ] Implement model version comparison
- [ ] Add support for drawing canvas input
- [ ] Create mobile app version

---

**Live Demo**: Try the web application by running `streamlit run main.py` after installation.

**Note**: This project focuses on individual character recognition. For complete text recognition, additional preprocessing for character segmentation would be needed.
