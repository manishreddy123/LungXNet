"""
Advanced Voting Classifier for Ensemble Learning
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import entropy
import warnings

class VotingClassifier:
    """Advanced voting classifier with multiple voting strategies"""
    
    def __init__(self, models: Dict[str, tf.keras.Model], 
                 weights: Optional[Dict[str, float]] = None,
                 voting_strategy: str = 'soft'):
        """
        Initialize voting classifier
        
        Args:
            models: Dictionary of model_name -> model
            weights: Dictionary of model_name -> weight (optional)
            voting_strategy: 'hard', 'soft', 'adaptive', or 'confidence_weighted'
        """
        self.models = models
        self.model_names = list(models.keys())
        self.num_models = len(models)
        self.voting_strategy = voting_strategy
        self.logger = logging.getLogger(__name__)
        
        # Initialize weights
        if weights is None:
            self.weights = {name: 1.0 / self.num_models for name in self.model_names}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            self.weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Validation
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input models and weights"""
        if not self.models:
            raise ValueError("At least one model must be provided")
        
        if set(self.weights.keys()) != set(self.model_names):
            raise ValueError("Weight keys must match model names")
        
        if self.voting_strategy not in ['hard', 'soft', 'adaptive', 'confidence_weighted']:
            raise ValueError("Invalid voting strategy")
    
    def predict(self, X, return_confidence: bool = False, 
                return_individual: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Make ensemble predictions
        
        Args:
            X: Input data
            return_confidence: Whether to return prediction confidence
            return_individual: Whether to return individual model predictions
            
        Returns:
            Predictions and optionally confidence scores and individual predictions
        """
        # Get predictions from all models
        individual_predictions = self._get_individual_predictions(X)
        
        # Apply voting strategy
        if self.voting_strategy == 'hard':
            ensemble_pred = self._hard_voting(individual_predictions)
        elif self.voting_strategy == 'soft':
            ensemble_pred = self._soft_voting(individual_predictions)
        elif self.voting_strategy == 'adaptive':
            ensemble_pred = self._adaptive_voting(individual_predictions, X)
        elif self.voting_strategy == 'confidence_weighted':
            ensemble_pred = self._confidence_weighted_voting(individual_predictions)
        
        results = [ensemble_pred]
        
        if return_confidence:
            confidence = self._calculate_confidence(ensemble_pred, individual_predictions)
            results.append(confidence)
        
        if return_individual:
            results.append(individual_predictions)
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def _get_individual_predictions(self, X) -> Dict[str, np.ndarray]:
        """Get predictions from all individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X, verbose=0)
                predictions[name] = pred
            except Exception as e:
                self.logger.error(f"Error getting predictions from model '{name}': {e}")
                # Create dummy predictions with uniform distribution
                num_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
                num_classes = self._infer_num_classes()
                dummy_pred = np.ones((num_samples, num_classes)) / num_classes
                predictions[name] = dummy_pred
        
        return predictions
    
    def _infer_num_classes(self) -> int:
        """Infer number of classes from models"""
        for model in self.models.values():
            try:
                return model.output_shape[-1]
            except:
                continue
        return 4  # Default for lung cancer classification
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard voting: majority vote of predicted classes"""
        # Convert probabilities to class predictions
        class_predictions = {}
        for name, pred in predictions.items():
            class_predictions[name] = np.argmax(pred, axis=1)
        
        # Weighted voting
        num_samples = len(next(iter(predictions.values())))
        num_classes = self._infer_num_classes()
        vote_counts = np.zeros((num_samples, num_classes))
        
        for name, class_pred in class_predictions.items():
            weight = self.weights[name]
            for i, cls in enumerate(class_pred):
                vote_counts[i, cls] += weight
        
        # Return class with most votes
        final_predictions = np.argmax(vote_counts, axis=1)
        
        # Convert to one-hot for consistency
        one_hot = np.zeros((num_samples, num_classes))
        one_hot[np.arange(num_samples), final_predictions] = 1.0
        
        return one_hot
    
    def _soft_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Soft voting: weighted average of predicted probabilities"""
        weighted_sum = None
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights[name]
            
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_sum /= total_weight
        
        return weighted_sum
    
    def _adaptive_voting(self, predictions: Dict[str, np.ndarray], X) -> np.ndarray:
        """Adaptive voting: adjust weights based on prediction confidence"""
        # Calculate confidence for each model's predictions
        model_confidences = {}
        
        for name, pred in predictions.items():
            # Use max probability as confidence measure
            confidences = np.max(pred, axis=1)
            model_confidences[name] = confidences
        
        # Adaptive weights based on confidence
        num_samples = len(next(iter(predictions.values())))
        adaptive_predictions = np.zeros_like(next(iter(predictions.values())))
        
        for i in range(num_samples):
            # Calculate adaptive weights for this sample
            sample_weights = {}
            total_confidence = 0
            
            for name in self.model_names:
                confidence = model_confidences[name][i]
                base_weight = self.weights[name]
                adaptive_weight = base_weight * confidence
                sample_weights[name] = adaptive_weight
                total_confidence += adaptive_weight
            
            # Normalize weights
            if total_confidence > 0:
                for name in sample_weights:
                    sample_weights[name] /= total_confidence
            
            # Weighted combination for this sample
            sample_pred = np.zeros(adaptive_predictions.shape[1])
            for name, pred in predictions.items():
                sample_pred += pred[i] * sample_weights[name]
            
            adaptive_predictions[i] = sample_pred
        
        return adaptive_predictions
    
    def _confidence_weighted_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Confidence-weighted voting using prediction entropy"""
        # Calculate entropy (uncertainty) for each prediction
        model_uncertainties = {}
        
        for name, pred in predictions.items():
            # Calculate entropy for each prediction
            entropies = np.array([entropy(p + 1e-10) for p in pred])
            # Convert entropy to confidence (lower entropy = higher confidence)
            max_entropy = np.log(pred.shape[1])  # Maximum possible entropy
            confidences = 1 - (entropies / max_entropy)
            model_uncertainties[name] = confidences
        
        # Weight predictions by confidence
        num_samples = len(next(iter(predictions.values())))
        confidence_weighted_pred = np.zeros_like(next(iter(predictions.values())))
        
        for i in range(num_samples):
            total_confidence = 0
            weighted_pred = np.zeros(confidence_weighted_pred.shape[1])
            
            for name, pred in predictions.items():
                confidence = model_uncertainties[name][i]
                base_weight = self.weights[name]
                final_weight = base_weight * confidence
                
                weighted_pred += pred[i] * final_weight
                total_confidence += final_weight
            
            # Normalize
            if total_confidence > 0:
                weighted_pred /= total_confidence
            
            confidence_weighted_pred[i] = weighted_pred
        
        return confidence_weighted_pred
    
    def _calculate_confidence(self, ensemble_pred: np.ndarray, 
                            individual_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate confidence scores for ensemble predictions"""
        
        # Method 1: Maximum probability
        max_prob_confidence = np.max(ensemble_pred, axis=1)
        
        # Method 2: Agreement between models
        agreement_scores = []
        for i in range(len(ensemble_pred)):
            # Get predicted classes from each model
            model_classes = []
            for pred in individual_predictions.values():
                model_classes.append(np.argmax(pred[i]))
            
            # Calculate agreement (most common prediction frequency)
            unique, counts = np.unique(model_classes, return_counts=True)
            max_agreement = np.max(counts) / len(model_classes)
            agreement_scores.append(max_agreement)
        
        agreement_confidence = np.array(agreement_scores)
        
        # Method 3: Entropy-based confidence
        entropy_confidence = []
        for pred in ensemble_pred:
            pred_entropy = entropy(pred + 1e-10)
            max_entropy = np.log(len(pred))
            confidence = 1 - (pred_entropy / max_entropy)
            entropy_confidence.append(confidence)
        
        entropy_confidence = np.array(entropy_confidence)
        
        # Combine confidence measures
        combined_confidence = (max_prob_confidence + agreement_confidence + entropy_confidence) / 3
        
        return combined_confidence
    
    def evaluate(self, X, y_true: np.ndarray, 
                return_detailed: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict]]:
        """
        Evaluate ensemble performance
        
        Args:
            X: Input data
            y_true: True labels
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Performance metrics and optionally detailed results
        """
        # Get ensemble predictions
        ensemble_pred, confidence, individual_pred = self.predict(
            X, return_confidence=True, return_individual=True
        )
        
        # Convert to class predictions
        y_pred = np.argmax(ensemble_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': np.mean(confidence)
        }
        
        if return_detailed:
            # Individual model performance
            individual_metrics = {}
            for name, pred in individual_pred.items():
                y_pred_individual = np.argmax(pred, axis=1)
                acc = accuracy_score(y_true, y_pred_individual)
                individual_metrics[name] = acc
            
            detailed_results = {
                'individual_accuracy': individual_metrics,
                'ensemble_predictions': ensemble_pred,
                'confidence_scores': confidence,
                'individual_predictions': individual_pred
            }
            
            return metrics, detailed_results
        
        return metrics
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights"""
        # Normalize weights
        total_weight = sum(new_weights.values())
        self.weights = {name: weight / total_weight for name, weight in new_weights.items()}
        
        self.logger.info(f"Updated ensemble weights: {self.weights}")
    
    def set_voting_strategy(self, strategy: str):
        """Change voting strategy"""
        if strategy not in ['hard', 'soft', 'adaptive', 'confidence_weighted']:
            raise ValueError("Invalid voting strategy")
        
        self.voting_strategy = strategy
        self.logger.info(f"Changed voting strategy to: {strategy}")
    
    def get_model_importance(self, X, y_true: np.ndarray) -> Dict[str, float]:
        """Calculate model importance based on individual performance"""
        importance_scores = {}
        
        # Get individual predictions
        individual_predictions = self._get_individual_predictions(X)
        
        for name, pred in individual_predictions.items():
            y_pred = np.argmax(pred, axis=1)
            accuracy = accuracy_score(y_true, y_pred)
            importance_scores[name] = accuracy
        
        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {name: score / total_importance 
                               for name, score in importance_scores.items()}
        
        return importance_scores
    
    def analyze_disagreement(self, X, threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze disagreement between models"""
        individual_predictions = self._get_individual_predictions(X)
        
        # Calculate disagreement metrics
        disagreement_samples = []
        agreement_scores = []
        
        num_samples = len(next(iter(individual_predictions.values())))
        
        for i in range(num_samples):
            # Get predicted classes from each model
            model_classes = []
            for pred in individual_predictions.values():
                model_classes.append(np.argmax(pred[i]))
            
            # Calculate agreement
            unique, counts = np.unique(model_classes, return_counts=True)
            max_agreement = np.max(counts) / len(model_classes)
            agreement_scores.append(max_agreement)
            
            # Mark samples with high disagreement
            if max_agreement < threshold:
                disagreement_samples.append(i)
        
        analysis = {
            'disagreement_samples': disagreement_samples,
            'agreement_scores': agreement_scores,
            'mean_agreement': np.mean(agreement_scores),
            'disagreement_rate': len(disagreement_samples) / num_samples
        }
        
        return analysis
    
    def get_summary(self) -> str:
        """Get voting classifier summary"""
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("VOTING CLASSIFIER SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Number of Models: {self.num_models}")
        summary_lines.append(f"Voting Strategy: {self.voting_strategy}")
        summary_lines.append("\nModel Weights:")
        
        for name, weight in self.weights.items():
            summary_lines.append(f"  {name}: {weight:.4f}")
        
        summary_lines.append("\nAvailable Strategies:")
        strategies = ['hard', 'soft', 'adaptive', 'confidence_weighted']
        for strategy in strategies:
            marker = " (current)" if strategy == self.voting_strategy else ""
            summary_lines.append(f"  - {strategy}{marker}")
        
        summary_lines.append("=" * 50)
        
        return "\n".join(summary_lines)
