"""
Advanced Metrics for Model Evaluation
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc, roc_curve, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
from typing import Dict, List, Tuple, Any, Optional
import logging
import pandas as pd
from scipy import stats

class AdvancedMetrics:
    """Comprehensive metrics calculation for medical image classification"""
    
    def __init__(self, class_names: List[str], num_classes: int):
        self.class_names = class_names
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics for model evaluation"""
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Advanced classification metrics
        metrics.update(self._calculate_advanced_metrics(y_true, y_pred))
        
        # Class-wise metrics
        metrics.update(self._calculate_classwise_metrics(y_true, y_pred))
        
        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_pred_proba))
        
        # Medical-specific metrics
        metrics.update(self._calculate_medical_metrics(y_true, y_pred))
        
        # Confusion matrix analysis
        metrics.update(self._analyze_confusion_matrix(y_true, y_pred))
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate advanced classification metrics"""
        
        return {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'top_1_accuracy': accuracy_score(y_true, y_pred),
        }
    
    def _calculate_classwise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics"""
        
        # Per-class precision, recall, f1-score
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        classwise_metrics = {}
        for i, class_name in enumerate(self.class_names):
            classwise_metrics[f'{class_name}_precision'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
            classwise_metrics[f'{class_name}_recall'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
            classwise_metrics[f'{class_name}_f1'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        return {'classwise_metrics': classwise_metrics}
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate probability-based metrics"""
        
        metrics = {}
        
        try:
            # Convert to one-hot if needed
            if len(y_true.shape) == 1:
                y_true_onehot = tf.keras.utils.to_categorical(y_true, self.num_classes)
            else:
                y_true_onehot = y_true
            
            # ROC AUC (macro and micro)
            if self.num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['roc_auc_macro'] = roc_auc_score(y_true_onehot, y_pred_proba, average='macro')
                metrics['roc_auc_micro'] = roc_auc_score(y_true_onehot, y_pred_proba, average='micro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_onehot, y_pred_proba, average='weighted')
            
            # Per-class ROC AUC
            roc_auc_per_class = {}
            for i, class_name in enumerate(self.class_names):
                if i < y_pred_proba.shape[1]:
                    try:
                        auc_score = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
                        roc_auc_per_class[f'{class_name}_roc_auc'] = auc_score
                    except ValueError:
                        roc_auc_per_class[f'{class_name}_roc_auc'] = 0.0
            
            metrics['roc_auc_per_class'] = roc_auc_per_class
            
            # Precision-Recall AUC
            pr_auc_per_class = {}
            for i, class_name in enumerate(self.class_names):
                if i < y_pred_proba.shape[1]:
                    try:
                        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
                        pr_auc = auc(recall, precision)
                        pr_auc_per_class[f'{class_name}_pr_auc'] = pr_auc
                    except ValueError:
                        pr_auc_per_class[f'{class_name}_pr_auc'] = 0.0
            
            metrics['pr_auc_per_class'] = pr_auc_per_class
            
            # Average Precision-Recall AUC
            metrics['pr_auc_macro'] = np.mean(list(pr_auc_per_class.values()))
            
            # Calibration metrics
            metrics.update(self._calculate_calibration_metrics(y_true, y_pred_proba))
            
        except Exception as e:
            self.logger.error(f"Error calculating probability metrics: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['pr_auc_macro'] = 0.0
        
        return metrics
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate model calibration metrics"""
        
        try:
            # Expected Calibration Error (ECE)
            ece = self._expected_calibration_error(y_true, y_pred_proba)
            
            # Maximum Calibration Error (MCE)
            mce = self._maximum_calibration_error(y_true, y_pred_proba)
            
            # Brier Score
            brier_score = self._brier_score(y_true, y_pred_proba)
            
            return {
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'brier_score': brier_score
            }
        except Exception as e:
            self.logger.error(f"Error calculating calibration metrics: {e}")
            return {
                'expected_calibration_error': 0.0,
                'maximum_calibration_error': 0.0,
                'brier_score': 0.0
            }
    
    def _expected_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        accuracies = (y_pred == y_true)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        accuracies = (y_pred == y_true)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _brier_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Brier Score"""
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(y_true, self.num_classes)
        else:
            y_true_onehot = y_true
        
        return np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    
    def _calculate_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate medical-specific metrics"""
        
        # Assuming class 1 is 'normal' and others are cancer types
        normal_class = 1 if 'normal' in [name.lower() for name in self.class_names] else 0
        
        # Binary classification: normal vs cancer
        y_true_binary = (y_true != normal_class).astype(int)
        y_pred_binary = (y_pred != normal_class).astype(int)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        
        # Positive and Negative Predictive Values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision for positive class
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Precision for negative class
        
        # Likelihood ratios
        lr_positive = sensitivity / (1 - specificity) if specificity < 1.0 else float('inf')
        lr_negative = (1 - sensitivity) / specificity if specificity > 0.0 else float('inf')
        
        # Diagnostic odds ratio
        dor = lr_positive / lr_negative if lr_negative > 0 else float('inf')
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'likelihood_ratio_positive': lr_positive,
            'likelihood_ratio_negative': lr_negative,
            'diagnostic_odds_ratio': dor
        }
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix for insights"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(cm):
                class_accuracies[f'{class_name}_accuracy'] = cm_normalized[i, i]
        
        # Most confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confusion_rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                    if confusion_rate > 0.1:  # More than 10% confusion
                        confused_pairs.append({
                            'true_class': self.class_names[i] if i < len(self.class_names) else f'Class_{i}',
                            'predicted_class': self.class_names[j] if j < len(self.class_names) else f'Class_{j}',
                            'confusion_rate': confusion_rate,
                            'count': cm[i, j]
                        })
        
        return {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'class_accuracies': class_accuracies,
            'confused_pairs': confused_pairs
        }
    
    def calculate_statistical_significance(self, y_true: np.ndarray, 
                                         predictions_1: np.ndarray, 
                                         predictions_2: np.ndarray) -> Dict[str, float]:
        """Calculate statistical significance between two models"""
        
        # McNemar's test for comparing two models
        # Create contingency table
        correct_1 = (y_true == predictions_1)
        correct_2 = (y_true == predictions_2)
        
        # Both correct, both wrong, model1 correct & model2 wrong, model1 wrong & model2 correct
        both_correct = np.sum(correct_1 & correct_2)
        both_wrong = np.sum(~correct_1 & ~correct_2)
        model1_correct_model2_wrong = np.sum(correct_1 & ~correct_2)
        model1_wrong_model2_correct = np.sum(~correct_1 & correct_2)
        
        # McNemar's test statistic
        if (model1_correct_model2_wrong + model1_wrong_model2_correct) > 0:
            mcnemar_statistic = ((abs(model1_correct_model2_wrong - model1_wrong_model2_correct) - 1) ** 2) / \
                               (model1_correct_model2_wrong + model1_wrong_model2_correct)
            mcnemar_p_value = 1 - stats.chi2.cdf(mcnemar_statistic, 1)
        else:
            mcnemar_statistic = 0.0
            mcnemar_p_value = 1.0
        
        # Paired t-test on accuracies (if we have multiple folds)
        acc_1 = accuracy_score(y_true, predictions_1)
        acc_2 = accuracy_score(y_true, predictions_2)
        
        return {
            'mcnemar_statistic': mcnemar_statistic,
            'mcnemar_p_value': mcnemar_p_value,
            'accuracy_difference': acc_2 - acc_1,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'model1_better': model1_correct_model2_wrong,
            'model2_better': model1_wrong_model2_correct
        }
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        
        return report
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics using bootstrap"""
        
        n_bootstrap = 1000
        n_samples = len(y_true)
        
        # Bootstrap sampling
        bootstrap_accuracies = []
        bootstrap_precisions = []
        bootstrap_recalls = []
        bootstrap_f1s = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            bootstrap_accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_precisions.append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            bootstrap_recalls.append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            bootstrap_f1s.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {
            'accuracy': (np.percentile(bootstrap_accuracies, lower_percentile),
                        np.percentile(bootstrap_accuracies, upper_percentile)),
            'precision': (np.percentile(bootstrap_precisions, lower_percentile),
                         np.percentile(bootstrap_precisions, upper_percentile)),
            'recall': (np.percentile(bootstrap_recalls, lower_percentile),
                      np.percentile(bootstrap_recalls, upper_percentile)),
            'f1_score': (np.percentile(bootstrap_f1s, lower_percentile),
                        np.percentile(bootstrap_f1s, upper_percentile))
        }
        
        return confidence_intervals
    
    def get_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted summary of all metrics"""
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("COMPREHENSIVE METRICS SUMMARY")
        summary_lines.append("=" * 80)
        
        # Basic metrics
        summary_lines.append("\nBASIC CLASSIFICATION METRICS:")
        summary_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        summary_lines.append(f"  Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
        summary_lines.append(f"  Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
        summary_lines.append(f"  F1-Score (Macro): {metrics.get('f1_macro', 0):.4f}")
        
        # Advanced metrics
        summary_lines.append("\nADVANCED METRICS:")
        summary_lines.append(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        summary_lines.append(f"  Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}")
        summary_lines.append(f"  Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.4f}")
        
        # Medical metrics
        summary_lines.append("\nMEDICAL METRICS:")
        summary_lines.append(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
        summary_lines.append(f"  Specificity: {metrics.get('specificity', 0):.4f}")
        summary_lines.append(f"  PPV: {metrics.get('positive_predictive_value', 0):.4f}")
        summary_lines.append(f"  NPV: {metrics.get('negative_predictive_value', 0):.4f}")
        
        # ROC AUC
        if 'roc_auc_macro' in metrics:
            summary_lines.append("\nROC AUC SCORES:")
            summary_lines.append(f"  Macro: {metrics.get('roc_auc_macro', 0):.4f}")
            summary_lines.append(f"  Micro: {metrics.get('roc_auc_micro', 0):.4f}")
        
        # Calibration
        summary_lines.append("\nCALIBRATION METRICS:")
        summary_lines.append(f"  Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}")
        summary_lines.append(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)
