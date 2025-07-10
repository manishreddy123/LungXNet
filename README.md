# LungXNet – Advanced Ensemble Deep Learning for Lung Cancer Detection

🚀 Welcome to LungXNet! This project leverages cutting-edge deep learning techniques to accurately detect lung cancer from CT scan images. 🩻✨

🧠 **What does this project do?**  
It combines multiple powerful neural network architectures to analyze medical images and classify them into different lung cancer types, helping in early diagnosis and treatment planning. The system is designed to be robust, accurate, and interpretable.

🏗️ **Architectures Used:**  
- **Capsule Networks** 🕸️: Capture spatial relationships and pose information for better feature understanding.  
- **U-Net with Skip Connections** 🔄: Enables detailed feature recovery for precise segmentation.  
- **Spatial Pyramid Pooling** 🏞️: Extracts multi-scale features to capture context at various resolutions.  
- **Advanced Pooling Techniques** 💧: Includes fractional max pooling and global average pooling for improved feature aggregation.  
- **Strided Convolutions** ➡️: Learnable downsampling to reduce spatial dimensions effectively.  
- **Ensemble Learning** 🤝: Combines predictions from multiple models to boost accuracy and robustness.

🎯 **Goal:**  
To classify CT scan images into four categories:  
1. Lung Squamous cell carcinoma  
2. Normal  
3. Lung Adenocarcinoma  
4. Large cell carcinoma  

This comprehensive approach ensures high performance and reliability in lung cancer detection, making it a valuable tool for medical professionals.

A comprehensive deep learning project for lung cancer detection using CT scan images, combining multiple state-of-the-art architectures and ensemble learning techniques.

## Project Overview

This project implements an advanced lung cancer detection system that combines the best features from multiple deep learning architectures:

- **Capsule Networks** for spatial relationship modeling
- **U-Net** with skip connections for detailed feature recovery
- **Spatial Pyramid Pooling** for multi-scale feature extraction
- **Advanced Pooling Techniques** (Fractional Max Pooling, Global Average Pooling)
- **Strided Convolutions** for learnable downsampling
- **Ensemble Learning** for improved accuracy and robustness

## Dataset

The system classifies CT scan images into 4 categories:
1. Lung Squamous cell carcinoma
2. Normal
3. Lung Adenocarcinoma
4. Large cell carcinoma

## Project Structure

```
best_project/
├── models/                 # Individual model implementations
│   ├── capsule_network.py
│   ├── unet_model.py
│   ├── hybrid_model.py
│   └── attention_model.py
├── ensemble/              # Ensemble learning framework
│   ├── ensemble_trainer.py
│   └── voting_classifier.py
├── utils/                 # Utility functions
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   └── visualization.py
├── evaluation/            # Model evaluation and metrics
│   ├── metrics.py
│   ├── comparison.py
│   └── reports.py
├── main.py               # Main execution script
├── config.py             # Configuration settings
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Features

### Advanced Architectures
- **Hybrid U-Net + Capsule Network**: Combines encoder-decoder architecture with capsule layers
- **Attention Mechanisms**: Self-attention and channel attention for better feature focus
- **Multi-Scale Feature Extraction**: Spatial pyramid pooling at multiple scales
- **Advanced Pooling**: Fractional max pooling and global average pooling

### Ensemble Learning
- **Model Diversity**: Multiple architectures with different strengths
- **Voting Strategies**: Hard and soft voting for final predictions
- **Weighted Ensemble**: Performance-based weight assignment

### Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Confusion Matrix**: Detailed classification analysis
- **Visualization**: Training curves, prediction confidence, feature maps
- **Model Comparison**: Side-by-side performance analysis

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run the Project

### Configuration

Before running the project, update the data paths in `config.py` to point to your local dataset folders:

```python
DATA_CONFIG = {
    # ... other settings ...
    'data_paths': {
        'train': "path/to/your/train/folder",
        'test': "path/to/your/test/folder",
        'val': "path/to/your/validation/folder"
    }
}
```

### Running the Demo

To test the system with a demo that includes model architectures, ensemble voting, visualization, and metrics:

```bash
python demo.py
```

### Training Models

Train individual models by specifying the model name and training parameters:

```bash
python main.py --mode train --model capsule_network --epochs 50
python main.py --mode train --model unet_model --epochs 50
python main.py --mode train --model attention_model --epochs 50
python main.py --mode train --model spatial_pyramid_model --epochs 50
```

Train the full ensemble model:

```bash
python main.py --mode train --model ensemble --epochs 100
```

### Evaluating Models

Evaluate individual models:

```bash
python main.py --mode evaluate --model capsule_network
python main.py --mode evaluate --model unet_model
```

Evaluate the ensemble model:

```bash
python main.py --mode evaluate --model ensemble
```

### Running the Complete Pipeline

Run the full training and evaluation pipeline, including ensemble training and evaluation:

```bash
python main.py --mode pipeline
```

## Model Performance

The ensemble system achieves superior performance by combining:
- Individual model strengths
- Diverse architectural approaches
- Advanced feature extraction techniques
- Robust evaluation metrics

## Configuration

Modify `config.py` to adjust:
- Model hyperparameters
- Training settings
- Data paths
- Evaluation metrics

## Contributing

This project demonstrates advanced deep learning techniques for medical image classification and can be extended with:
- Additional architectures
- New ensemble strategies
- Enhanced preprocessing techniques
- Advanced visualization methods
