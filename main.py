"""
Main execution script for Advanced Lung Cancer Detection System
"""

import os
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import DataLoader, ModelVisualizer
from models.efficient_models import (EfficientCapsuleNetwork, EfficientUNet, EfficientHybridModel, 
                                    EfficientAttentionModel, EfficientSpatialPyramidModel)
from models import (CapsuleNetwork, UNetModel, HybridUNetCapsule, 
                   AttentionModel, SpatialPyramidModel)
from ensemble import EnsembleTrainer, VotingClassifier
from evaluation import AdvancedMetrics

class LungCancerDetectionSystem:
    """Main system for lung cancer detection"""
    
    def __init__(self, config_override=None):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.data_config = DATA_CONFIG
        self.training_config = TRAINING_CONFIG
        self.model_config = MODEL_CONFIG
        self.ensemble_config = ENSEMBLE_CONFIG
        
        if config_override:
            self.update_config(config_override)
        
        # Setup GPU
        self.setup_gpu()
        
        # Initialize components
        self.data_loader = DataLoader(self.data_config)
        self.visualizer = ModelVisualizer(self.data_config['class_names'])
        self.metrics_calculator = AdvancedMetrics(
            self.data_config['class_names'], 
            self.data_config['num_classes']
        )
        
        # Data placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        self.logger.info("Lung Cancer Detection System initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_gpu(self):
        """Setup GPU configuration"""
        if HARDWARE_CONFIG['use_gpu']:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    if HARDWARE_CONFIG['mixed_precision']:
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    
                    self.logger.info(f"GPU setup complete. Found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    self.logger.error(f"GPU setup error: {e}")
            else:
                self.logger.warning("No GPUs found, using CPU")
        else:
            self.logger.info("Using CPU for computation")
    
    def update_config(self, config_override):
        """Update configuration with override values"""
        for key, value in config_override.items():
            if hasattr(self, f"{key}_config"):
                getattr(self, f"{key}_config").update(value)
    
    def load_data(self):
        """Load and preprocess data"""
        self.logger.info("Loading datasets...")
        
        # Print dataset information
        self.data_loader.print_dataset_info()
        
        # Load datasets
        self.train_ds, self.val_ds, self.test_ds = self.data_loader.load_all_datasets()
        
        if self.train_ds is None:
            raise ValueError("Failed to load training dataset")
        
        # Apply preprocessing
        self.train_ds = self.data_loader.preprocess_dataset(self.train_ds, apply_augmentation=True)
        self.val_ds = self.data_loader.preprocess_dataset(self.val_ds, apply_augmentation=False)
        self.test_ds = self.data_loader.preprocess_dataset(self.test_ds, apply_augmentation=False)
        
        self.logger.info("Datasets loaded and preprocessed successfully")
    
    def train_individual_model(self, model_name: str, model_config: dict = None):
        """Train an individual model"""
        self.logger.info(f"Training {model_name}...")
        
        if model_config is None:
            model_config = self.model_config.get(model_name, {})
        
        input_shape = (self.data_config['image_size'], 
                      self.data_config['image_size'], 
                      self.data_config['channels'])
        num_classes = self.data_config['num_classes']
        
        # Initialize model
        if model_name == 'capsule_network':
            model_builder = CapsuleNetwork(input_shape, num_classes, model_config)
            model = model_builder.create_simple_capsnet()
        elif model_name == 'unet_model':
            model_builder = UNetModel(input_shape, num_classes, model_config)
            model = model_builder.build_model()
        elif model_name == 'hybrid_unet_capsule':
            model_builder = HybridUNetCapsule(input_shape, num_classes, model_config)
            model = model_builder.build_model()
        elif model_name == 'attention_model':
            model_builder = AttentionModel(input_shape, num_classes, model_config)
            model = model_builder.build_lightweight_attention_model()
        elif model_name == 'spatial_pyramid_model':
            model_builder = SpatialPyramidModel(input_shape, num_classes, model_config)
            model = model_builder.build_model()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Compile model
        model_builder.compile_model(model, learning_rate=self.training_config['learning_rate'])
        
        # Print model summary
        print(model_builder.get_model_summary(model))
        
        # Setup callbacks
        callbacks = self._create_callbacks(model_name)
        
        # Train model
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(PATHS_CONFIG['models_dir'], f'{model_name}.h5')
        model.save(model_path)
        
        # Save training history
        history_path = os.path.join(PATHS_CONFIG['results_dir'], f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f, default=str)
        
        self.logger.info(f"Model {model_name} trained and saved successfully")
        
        return model, history.history
    
    def train_ensemble(self):
        """Train ensemble of models"""
        self.logger.info("Training ensemble models...")
        
        # Configuration for individual models
        model_configs = {
            'capsule_network': self.model_config.get('capsule_network', {}),
            'unet_model': self.model_config.get('unet', {}),
            'attention_model': self.model_config.get('attention', {}),
            'spatial_pyramid_model': self.model_config.get('spatial_pyramid', {}),
        }
        
        # Initialize ensemble trainer
        ensemble_trainer = EnsembleTrainer({
            'data': self.data_config,
            'training': self.training_config,
            'models_dir': PATHS_CONFIG['models_dir']
        })
        
        # Train individual models
        trained_models = ensemble_trainer.train_individual_models(
            self.train_ds, self.val_ds, model_configs
        )
        
        # Add models to ensemble
        for name, model_info in trained_models.items():
            ensemble_trainer.add_model(name, model_info['model'])
        
        # Calculate optimal weights
        ensemble_trainer.calculate_model_weights(self.val_ds)
        
        # Create stacking ensemble
        if self.ensemble_config.get('use_stacking', False):
            ensemble_trainer.create_stacking_ensemble(
                self.train_ds, self.val_ds,
                meta_learner_type=self.ensemble_config.get('meta_learner', 'random_forest')
            )
        
        # Save ensemble
        ensemble_path = os.path.join(PATHS_CONFIG['models_dir'], 'ensemble.pkl')
        ensemble_trainer.save_ensemble(ensemble_path)
        
        self.logger.info("Ensemble training completed")
        
        return ensemble_trainer
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a single model"""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        predictions = model.predict(self.test_ds, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = np.concatenate([y for x, y in self.test_ds], axis=0)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, predictions)
        
        # Generate classification report
        report = self.metrics_calculator.generate_classification_report(y_true, y_pred)
        
        # Save results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(PATHS_CONFIG['results_dir'], f'{model_name}_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print(self.metrics_calculator.get_metrics_summary(metrics))
        
        return metrics
    
    def evaluate_ensemble(self, ensemble_trainer):
        """Evaluate ensemble models"""
        self.logger.info("Evaluating ensemble...")
        
        # Evaluate different ensemble methods
        ensemble_results = ensemble_trainer.evaluate_ensemble(
            self.test_ds, 
            methods=['weighted_voting', 'average', 'stacking']
        )
        
        # Print ensemble summary
        print(ensemble_trainer.get_ensemble_summary())
        
        # Save ensemble results
        results_path = os.path.join(PATHS_CONFIG['results_dir'], 'ensemble_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(ensemble_results, f, indent=2, default=str)
        
        return ensemble_results
    
    def generate_visualizations(self, model, model_name: str):
        """Generate visualizations for model analysis"""
        self.logger.info(f"Generating visualizations for {model_name}...")
        
        # Get sample data
        sample_images, sample_labels = self.data_loader.get_sample_batch(self.test_ds)
        
        if sample_images is not None:
            # Get predictions
            predictions = model.predict(sample_images, verbose=0)
            
            # Plot sample predictions
            self.visualizer.plot_sample_images(
                sample_images, sample_labels, predictions,
                save_path=os.path.join(PATHS_CONFIG['plots_dir'], f'{model_name}_predictions.png')
            )
        
        self.logger.info(f"Visualizations saved for {model_name}")
    
    def _create_callbacks(self, model_name: str):
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.training_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.training_config['reduce_lr_factor'],
            patience=self.training_config['reduce_lr_patience'],
            min_lr=self.training_config['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = os.path.join(PATHS_CONFIG['checkpoints_dir'], f'{model_name}_best.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        self.logger.info("Starting complete pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Train ensemble
            ensemble_trainer = self.train_ensemble()
            
            # Evaluate ensemble
            ensemble_results = self.evaluate_ensemble(ensemble_trainer)
            
            # Generate final report
            self._generate_final_report(ensemble_results)
            
            self.logger.info("Complete pipeline finished successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _generate_final_report(self, ensemble_results):
        """Generate final comprehensive report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LUNG CANCER DETECTION SYSTEM - FINAL REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("DATASET INFORMATION:")
        report_lines.append(f"  Classes: {', '.join(self.data_config['class_names'])}")
        report_lines.append(f"  Image Size: {self.data_config['image_size']}x{self.data_config['image_size']}")
        report_lines.append(f"  Channels: {self.data_config['channels']}")
        report_lines.append("")
        
        # Ensemble results
        report_lines.append("ENSEMBLE RESULTS:")
        for method, results in ensemble_results.items():
            report_lines.append(f"  {method.upper()}:")
            report_lines.append(f"    Accuracy: {results['accuracy']:.4f}")
            report_lines.append(f"    Precision: {results['precision']:.4f}")
            report_lines.append(f"    Recall: {results['recall']:.4f}")
            report_lines.append(f"    F1-Score: {results['f1_score']:.4f}")
            report_lines.append("")
        
        # Best performing method
        best_method = max(ensemble_results.keys(), 
                         key=lambda x: ensemble_results[x]['accuracy'])
        best_accuracy = ensemble_results[best_method]['accuracy']
        
        report_lines.append(f"BEST PERFORMING METHOD: {best_method.upper()}")
        report_lines.append(f"BEST ACCURACY: {best_accuracy:.4f}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Save report
        report_path = os.path.join(PATHS_CONFIG['results_dir'], 'final_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print report
        print('\n'.join(report_lines))

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Advanced Lung Cancer Detection System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'pipeline'], 
                       default='pipeline', help='Execution mode')
    parser.add_argument('--model', choices=['capsule_network', 'unet_model', 
                                          'hybrid_unet_capsule', 'attention_model', 
                                          'spatial_pyramid_model', 'ensemble'], 
                       default='ensemble', help='Model to train/evaluate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config override if provided
    config_override = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_override = json.load(f)
    
    # Override with command line arguments
    if args.epochs:
        config_override.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config_override.setdefault('data', {})['batch_size'] = args.batch_size
    if args.learning_rate:
        config_override.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    # Initialize system
    system = LungCancerDetectionSystem(config_override)
    
    try:
        if args.mode == 'pipeline':
            system.run_complete_pipeline()
        elif args.mode == 'train':
            system.load_data()
            if args.model == 'ensemble':
                system.train_ensemble()
            else:
                model, history = system.train_individual_model(args.model)
                system.generate_visualizations(model, args.model)
        elif args.mode == 'evaluate':
            system.load_data()
            if args.model == 'ensemble':
                # Load and evaluate ensemble
                ensemble_trainer = EnsembleTrainer({
                    'data': system.data_config,
                    'models_dir': PATHS_CONFIG['models_dir']
                })
                ensemble_path = os.path.join(PATHS_CONFIG['models_dir'], 'ensemble.pkl')
                ensemble_trainer.load_ensemble(ensemble_path)
                system.evaluate_ensemble(ensemble_trainer)
            else:
                # Load and evaluate individual model
                model_path = os.path.join(PATHS_CONFIG['models_dir'], f'{args.model}.h5')
                model = tf.keras.models.load_model(model_path)
                system.evaluate_model(model, args.model)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
