"""
Demo script for Advanced Lung Cancer Detection System
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import DataLoader, ModelVisualizer
from models import CapsuleNetwork, UNetModel
from ensemble import VotingClassifier

def create_demo_data():
    """Create demo data for testing (when real data is not available)"""
    print("Creating demo data for testing...")
    
    # Create synthetic data
    num_samples = 100
    image_size = DATA_CONFIG['image_size']
    channels = DATA_CONFIG['channels']
    num_classes = DATA_CONFIG['num_classes']
    
    # Generate random images
    X = np.random.rand(num_samples, image_size, image_size, channels) * 255
    X = X.astype(np.uint8)
    
    # Generate random labels
    y = np.random.randint(0, num_classes, num_samples)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(DATA_CONFIG['batch_size'])
    
    return dataset

def demo_individual_models():
    """Demonstrate individual model architectures"""
    print("\n" + "="*60)
    print("DEMONSTRATING INDIVIDUAL MODEL ARCHITECTURES")
    print("="*60)
    
    input_shape = (DATA_CONFIG['image_size'], DATA_CONFIG['image_size'], DATA_CONFIG['channels'])
    num_classes = DATA_CONFIG['num_classes']
    
    # Demo Capsule Network
    print("\n1. CAPSULE NETWORK")
    print("-" * 30)
    capsule_model = CapsuleNetwork(input_shape, num_classes, MODEL_CONFIG['capsule_network'])
    model = capsule_model.create_simple_capsnet()
    print(f"Model created: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Demo U-Net Model
    print("\n2. U-NET MODEL")
    print("-" * 30)
    unet_model = UNetModel(input_shape, num_classes, MODEL_CONFIG['unet'])
    model = unet_model.build_lightweight_unet()
    print(f"Model created: {model.name}")
    print(f"Total parameters: {model.count_params():,}")

def demo_ensemble_voting():
    """Demonstrate ensemble voting mechanisms"""
    print("\n" + "="*60)
    print("DEMONSTRATING ENSEMBLE VOTING")
    print("="*60)
    
    # Create simple models for demo
    input_shape = (DATA_CONFIG['image_size'], DATA_CONFIG['image_size'], DATA_CONFIG['channels'])
    num_classes = DATA_CONFIG['num_classes']
    
    models = {}
    
    # Create lightweight models
    for i, name in enumerate(['model_1', 'model_2', 'model_3']):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ], name=name)
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        models[name] = model
        print(f"Created {name}: {model.count_params():,} parameters")
    
    # Create voting classifier
    voting_classifier = VotingClassifier(models, voting_strategy='soft')
    print(f"\nVoting Classifier Summary:")
    print(voting_classifier.get_summary())
    
    # Demo with synthetic data
    demo_data = create_demo_data()
    sample_batch = next(iter(demo_data))
    X_sample, y_sample = sample_batch
    
    print(f"\nTesting with sample batch: {X_sample.shape}")
    
    # Test different voting strategies
    strategies = ['soft', 'hard', 'confidence_weighted']
    for strategy in strategies:
        voting_classifier.set_voting_strategy(strategy)
        predictions = voting_classifier.predict(X_sample)
        print(f"{strategy.upper()} voting - Prediction shape: {predictions.shape}")

def demo_data_loading():
    """Demonstrate data loading capabilities"""
    print("\n" + "="*60)
    print("DEMONSTRATING DATA LOADING")
    print("="*60)
    
    # Initialize data loader
    data_loader = DataLoader(DATA_CONFIG)
    
    # Print configuration
    print("Data Configuration:")
    print(f"  Image Size: {DATA_CONFIG['image_size']}x{DATA_CONFIG['image_size']}")
    print(f"  Batch Size: {DATA_CONFIG['batch_size']}")
    print(f"  Channels: {DATA_CONFIG['channels']}")
    print(f"  Classes: {DATA_CONFIG['class_names']}")
    
    # Try to load real data, fallback to demo data
    try:
        print("\nAttempting to load real datasets...")
        train_ds, val_ds, test_ds = data_loader.load_all_datasets()
        
        if train_ds is not None:
            print("✓ Real datasets loaded successfully")
            
            # Get sample batch
            sample_images, sample_labels = data_loader.get_sample_batch(train_ds)
            if sample_images is not None:
                print(f"Sample batch shape: {sample_images.shape}")
                print(f"Sample labels shape: {sample_labels.shape}")
        else:
            raise ValueError("No real data available")
            
    except Exception as e:
        print(f"✗ Could not load real data: {e}")
        print("Using demo data instead...")
        
        demo_dataset = create_demo_data()
        print(f"✓ Demo dataset created")
        
        # Show dataset info
        for batch in demo_dataset.take(1):
            images, labels = batch
            print(f"Demo batch shape: {images.shape}")
            print(f"Demo labels shape: {labels.shape}")

def load_model_weights(model_class, model_name, input_shape, num_classes, config):
    """
    Load a trained model from saved_models directory.
    """
    import tensorflow as tf
    model_instance = model_class(input_shape, num_classes, config)
    model = None
    try:
        model = model_instance.load_model(f"saved_models/{model_name}.h5")
        print(f"Loaded model weights for {model_name}")
    except Exception as e:
        print(f"Could not load model weights for {model_name}: {e}")
        model = model_instance.create_simple_capsnet() if hasattr(model_instance, 'create_simple_capsnet') else None
    return model

def demo_visualization():
    """Demonstrate visualization capabilities with real model predictions"""
    import os
    print("\n" + "="*60)
    print("DEMONSTRATING VISUALIZATION")
    print("="*60)
    
    # Initialize visualizer
    visualizer = ModelVisualizer(DATA_CONFIG['class_names'])
    
    # Load test dataset
    try:
        data_loader = DataLoader(DATA_CONFIG)
        _, _, test_ds = data_loader.load_all_datasets()
        if test_ds is None:
            raise ValueError("No test dataset available")
    except Exception as e:
        print(f"✗ Could not load test dataset for visualization: {e}")
        print("Using synthetic data instead...")
        test_ds = create_demo_data()
    
    # Load model for prediction
    input_shape = (DATA_CONFIG['image_size'], DATA_CONFIG['image_size'], DATA_CONFIG['channels'])
    num_classes = DATA_CONFIG['num_classes']
    model = load_model_weights(CapsuleNetwork, 'efficient_capsule_network', input_shape, num_classes, MODEL_CONFIG['capsule_network'])
    if model is None:
        print("No model available for prediction, skipping visualization.")
        return
    
    # Get a batch of test images and labels
    sample_batch = next(iter(test_ds))
    images, labels = sample_batch
    sample_images = images[:9].numpy()
    sample_labels = labels[:9].numpy()
    
    # Predict on sample images
    predictions = model.predict(sample_images)
    
    # Normalize predictions if needed
    if predictions.ndim == 2:
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
    
    print(f"Visualizing {len(sample_images)} sample images with model predictions...")
    
    try:
        save_path = os.path.join(PATHS_CONFIG['plots_dir'], 'demo_samples.png')
        visualizer.plot_sample_images(
            sample_images, sample_labels, predictions,
            save_path=save_path
        )
        print(f"✓ Sample visualization completed. Image saved to: {save_path}")
        
        # Attempt to open the saved image automatically (platform dependent)
        import platform
        import subprocess
        
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', save_path))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(save_path)
        elif platform.system() == 'Linux':      # Linux
            subprocess.call(('xdg-open', save_path))
        else:
            print("Automatic image opening not supported on this OS.")
        
    except Exception as e:
        print(f"✗ Visualization error: {e}")

def demo_metrics():
    """Demonstrate metrics calculation"""
    print("\n" + "="*60)
    print("DEMONSTRATING METRICS CALCULATION")
    print("="*60)
    
    from evaluation.metrics import AdvancedMetrics
    
    # Initialize metrics calculator
    metrics_calc = AdvancedMetrics(DATA_CONFIG['class_names'], DATA_CONFIG['num_classes'])
    
    # Create sample predictions
    num_samples = 100
    num_classes = DATA_CONFIG['num_classes']
    
    # Generate realistic predictions (with some accuracy)
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Create predictions with 70% accuracy
    y_pred = y_true.copy()
    wrong_indices = np.random.choice(num_samples, size=int(0.3 * num_samples), replace=False)
    y_pred[wrong_indices] = np.random.randint(0, num_classes, len(wrong_indices))
    
    # Create probability predictions
    y_pred_proba = np.random.rand(num_samples, num_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    print(f"Calculating metrics for {num_samples} samples...")
    
    # Calculate comprehensive metrics
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    # Print summary
    print(metrics_calc.get_metrics_summary(metrics))
    
    # Generate classification report
    report = metrics_calc.generate_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

def main():
    """Main demo function"""
    print("🚀 ADVANCED LUNG CANCER DETECTION SYSTEM - DEMO")
    print("=" * 80)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
    else:
        print("GPU not available, using CPU")
    
    try:
        # Run demos
        demo_data_loading()
        demo_individual_models()
        demo_ensemble_voting()
        demo_visualization()
        demo_metrics()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Update data paths in config.py")
        print("3. Run training: python main.py --mode train")
        print("4. Run full pipeline: python main.py --mode pipeline")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the requirements and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
