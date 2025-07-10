"""
Advanced Ensemble Training Framework
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class EnsembleTrainer:
    """Advanced ensemble training with multiple strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensemble_config = config.get('ensemble', {})
        self.models = {}
        self.model_weights = {}
        self.meta_learner = None
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.models_dir = config.get('models_dir', 'saved_models')
        self.ensemble_dir = os.path.join(self.models_dir, 'ensemble')
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
    def add_model(self, name: str, model: tf.keras.Model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.model_weights[name] = weight
        self.logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def train_individual_models(self, train_data, val_data, model_configs: Dict[str, Dict]):
        """Train individual models for the ensemble"""
        
        from ..models import (CapsuleNetwork, UNetModel, HybridUNetCapsule, 
                             AttentionModel, SpatialPyramidModel)
        
        input_shape = self.config['data']['input_shape']
        num_classes = self.config['data']['num_classes']
        
        model_classes = {
            'capsule_network': CapsuleNetwork,
            'unet_model': UNetModel,
            'hybrid_unet_capsule': HybridUNetCapsule,
            'attention_model': AttentionModel,
            'spatial_pyramid_model': SpatialPyramidModel
        }
        
        trained_models = {}
        
        for model_name, model_config in model_configs.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Initialize model
                model_class = model_classes[model_name]
                model_builder = model_class(input_shape, num_classes, model_config)
                
                # Build and compile model
                if model_name == 'capsule_network':
                    model = model_builder.create_simple_capsnet()
                elif model_name == 'hybrid_unet_capsule':
                    model = model_builder.build_model()
                elif model_name == 'attention_model':
                    model = model_builder.build_lightweight_attention_model()
                else:
                    model = model_builder.build_model()
                
                model_builder.compile_model(model, learning_rate=0.001)
                
                # Training callbacks
                callbacks = self._create_callbacks(model_name)
                
                # Train model
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=self.config['training']['epochs'],
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save model
                model_path = os.path.join(self.ensemble_dir, f'{model_name}.h5')
                model.save(model_path)
                
                # Save training history
                history_path = os.path.join(self.ensemble_dir, f'{model_name}_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)
                
                trained_models[model_name] = {
                    'model': model,
                    'history': history.history,
                    'path': model_path
                }
                
                self.logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        return trained_models
    
    def _create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks for individual models"""
        
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['training'].get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training'].get('reduce_lr_factor', 0.5),
            patience=self.config['training'].get('reduce_lr_patience', 10),
            min_lr=self.config['training'].get('min_lr', 1e-7),
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.ensemble_dir, f'{model_name}_best.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def calculate_model_weights(self, val_data) -> Dict[str, float]:
        """Calculate optimal weights for ensemble models based on validation performance"""
        
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Evaluate model on validation data
                val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
                model_scores[name] = val_accuracy
                self.logger.info(f"Model '{name}' validation accuracy: {val_accuracy:.4f}")
            except Exception as e:
                self.logger.error(f"Error evaluating model '{name}': {e}")
                model_scores[name] = 0.0
        
        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in model_scores.items()}
        else:
            # Equal weights if all models failed
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        self.model_weights = weights
        self.logger.info(f"Calculated ensemble weights: {weights}")
        
        return weights
    
    def create_stacking_ensemble(self, train_data, val_data, meta_learner_type: str = 'random_forest'):
        """Create stacking ensemble with meta-learner"""
        
        # Generate predictions from base models
        train_predictions = self._get_ensemble_predictions(train_data, training=True)
        val_predictions = self._get_ensemble_predictions(val_data, training=False)
        
        # Get true labels
        train_labels = np.concatenate([y for x, y in train_data], axis=0)
        val_labels = np.concatenate([y for x, y in val_data], axis=0)
        
        # Train meta-learner
        if meta_learner_type == 'random_forest':
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif meta_learner_type == 'logistic_regression':
            self.meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported meta-learner type: {meta_learner_type}")
        
        # Train meta-learner
        self.meta_learner.fit(train_predictions, train_labels)
        
        # Evaluate meta-learner
        val_pred = self.meta_learner.predict(val_predictions)
        accuracy = accuracy_score(val_labels, val_pred)
        
        self.logger.info(f"Stacking ensemble validation accuracy: {accuracy:.4f}")
        
        # Save meta-learner
        meta_learner_path = os.path.join(self.ensemble_dir, 'meta_learner.pkl')
        joblib.dump(self.meta_learner, meta_learner_path)
        
        return accuracy
    
    def _get_ensemble_predictions(self, data, training: bool = False) -> np.ndarray:
        """Get predictions from all ensemble models"""
        
        all_predictions = []
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(data, verbose=0)
                all_predictions.append(predictions)
            except Exception as e:
                self.logger.error(f"Error getting predictions from model '{name}': {e}")
                # Create dummy predictions
                dummy_pred = np.zeros((len(data), self.config['data']['num_classes']))
                all_predictions.append(dummy_pred)
        
        # Concatenate predictions from all models
        ensemble_predictions = np.concatenate(all_predictions, axis=1)
        
        return ensemble_predictions
    
    def predict_ensemble(self, data, method: str = 'weighted_voting') -> np.ndarray:
        """Make ensemble predictions using specified method"""
        
        if method == 'weighted_voting':
            return self._weighted_voting_predict(data)
        elif method == 'stacking':
            return self._stacking_predict(data)
        elif method == 'average':
            return self._average_predict(data)
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
    
    def _weighted_voting_predict(self, data) -> np.ndarray:
        """Weighted voting ensemble prediction"""
        
        weighted_predictions = None
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(data, verbose=0)
                weight = self.model_weights.get(name, 1.0)
                
                if weighted_predictions is None:
                    weighted_predictions = predictions * weight
                else:
                    weighted_predictions += predictions * weight
                
                total_weight += weight
                
            except Exception as e:
                self.logger.error(f"Error in weighted voting for model '{name}': {e}")
                continue
        
        if total_weight > 0:
            weighted_predictions /= total_weight
        
        return weighted_predictions
    
    def _stacking_predict(self, data) -> np.ndarray:
        """Stacking ensemble prediction using meta-learner"""
        
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call create_stacking_ensemble first.")
        
        # Get base model predictions
        base_predictions = self._get_ensemble_predictions(data)
        
        # Use meta-learner for final prediction
        final_predictions = self.meta_learner.predict_proba(base_predictions)
        
        return final_predictions
    
    def _average_predict(self, data) -> np.ndarray:
        """Simple average ensemble prediction"""
        
        all_predictions = []
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(data, verbose=0)
                all_predictions.append(predictions)
            except Exception as e:
                self.logger.error(f"Error in average prediction for model '{name}': {e}")
                continue
        
        if all_predictions:
            average_predictions = np.mean(all_predictions, axis=0)
            return average_predictions
        else:
            raise ValueError("No valid predictions from ensemble models")
    
    def evaluate_ensemble(self, test_data, methods: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate ensemble using different methods"""
        
        if methods is None:
            methods = ['weighted_voting', 'average']
            if self.meta_learner is not None:
                methods.append('stacking')
        
        results = {}
        
        # Get true labels
        true_labels = np.concatenate([y for x, y in test_data], axis=0)
        
        for method in methods:
            try:
                # Get ensemble predictions
                predictions = self.predict_ensemble(test_data, method=method)
                predicted_labels = np.argmax(predictions, axis=1)
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, predicted_labels)
                
                # Classification report
                report = classification_report(
                    true_labels, 
                    predicted_labels, 
                    output_dict=True,
                    zero_division=0
                )
                
                results[method] = {
                    'accuracy': accuracy,
                    'precision': report['macro avg']['precision'],
                    'recall': report['macro avg']['recall'],
                    'f1_score': report['macro avg']['f1-score']
                }
                
                self.logger.info(f"Ensemble method '{method}' - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating ensemble method '{method}': {e}")
                results[method] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        return results
    
    def save_ensemble(self, filepath: str):
        """Save the complete ensemble"""
        
        ensemble_data = {
            'model_weights': self.model_weights,
            'config': self.config,
            'model_paths': {name: os.path.join(self.ensemble_dir, f'{name}.h5') 
                           for name in self.models.keys()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a saved ensemble"""
        
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.model_weights = ensemble_data['model_weights']
        self.config = ensemble_data['config']
        
        # Load individual models
        self.models = {}
        for name, model_path in ensemble_data['model_paths'].items():
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    self.models[name] = model
                    self.logger.info(f"Loaded model '{name}' from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading model '{name}': {e}")
        
        # Load meta-learner if exists
        meta_learner_path = os.path.join(self.ensemble_dir, 'meta_learner.pkl')
        if os.path.exists(meta_learner_path):
            try:
                self.meta_learner = joblib.load(meta_learner_path)
                self.logger.info("Loaded meta-learner")
            except Exception as e:
                self.logger.error(f"Error loading meta-learner: {e}")
        
        self.logger.info(f"Ensemble loaded from {filepath}")
    
    def get_ensemble_summary(self) -> str:
        """Get detailed ensemble summary"""
        
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("ENSEMBLE SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Number of Models: {len(self.models)}")
        summary_lines.append(f"Meta-learner: {'Yes' if self.meta_learner else 'No'}")
        summary_lines.append("\nModel Weights:")
        
        for name, weight in self.model_weights.items():
            summary_lines.append(f"  {name}: {weight:.4f}")
        
        summary_lines.append("\nAvailable Methods:")
        methods = ['weighted_voting', 'average']
        if self.meta_learner:
            methods.append('stacking')
        
        for method in methods:
            summary_lines.append(f"  - {method}")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
