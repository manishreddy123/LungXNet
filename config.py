"""
Configuration file for the Advanced Lung Cancer Detection System
"""

import os

# Data Configuration
DATA_CONFIG = {
    'image_size': 256,
    'batch_size': 32,
    'channels': 1,
    'num_classes': 4,
    'class_names': [
        'Lung Squamous cell carcinoma',
        'normal', 
        'Lung Adenocarcinoma',
        'Large cell carcinoma'
    ],
    'data_paths': {
        'train': "C:/Users/manis/Documents/MIT/6th semester/DL/project/ct images/Data/train",
        'test': "C:/Users/manis/Documents/MIT/6th semester/DL/project/ct images/Data/test",
        'val': "C:/Users/manis/Documents/MIT/6th semester/DL/project/ct images/Data/valid"
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],
    'validation_split': 0.2,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}

# Model Configuration
MODEL_CONFIG = {
    'capsule_network': {
        'primary_caps_dim': 8,
        'primary_caps_channels': 32,
        'digit_caps_dim': 16,
        'routings': 3
    },
    'unet': {
        'filters': [64, 128, 256, 512],
        'dropout_rate': 0.3,
        'batch_norm': True
    },
    'attention': {
        'attention_dim': 256,
        'use_self_attention': True,
        'use_channel_attention': True
    },
    'spatial_pyramid': {
        'pool_sizes': [1, 2, 4, 8],
        'pool_type': 'max'
    }
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'models': ['hybrid_unet_capsule', 'attention_model', 'spatial_pyramid_model'],
    'voting_strategy': 'soft',  # 'hard' or 'soft'
    'weights': None,  # Auto-calculate based on validation performance
    'use_stacking': True,
    'meta_learner': 'random_forest'
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 20,
    'zoom_range': 0.2,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'brightness_range': [0.8, 1.2],
    'contrast_range': [0.8, 1.2],
    'noise_factor': 0.1
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'auc_roc', 'auc_pr', 'confusion_matrix'
    ],
    'plot_training_curves': True,
    'plot_confusion_matrix': True,
    'plot_roc_curves': True,
    'plot_precision_recall_curves': True,
    'save_predictions': True,
    'generate_classification_report': True
}

# Paths Configuration
PATHS_CONFIG = {
    'models_dir': 'saved_models',
    'logs_dir': 'logs',
    'results_dir': 'results',
    'plots_dir': 'plots',
    'checkpoints_dir': 'checkpoints'
}

# Create directories if they don't exist
for path in PATHS_CONFIG.values():
    os.makedirs(path, exist_ok=True)

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
    'mixed_precision': True,
    'parallel_model': False
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/training.log',
    'tensorboard_logs': 'logs/tensorboard',
    'save_model_history': True
}
