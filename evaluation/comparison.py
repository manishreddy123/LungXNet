"""
Evaluation Comparison Module for Advanced Lung Cancer Detection System

Provides utilities for comparing multiple models' performance,
generating detailed reports, and visualizing comparison results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ModelComparison:
    def __init__(self, model_names, y_true, y_preds, y_pred_probas=None, class_names=None):
        """
        Initialize the ModelComparison object.

        Parameters:
        - model_names: list of str, names of the models being compared
        - y_true: array-like, true labels
        - y_preds: dict, keys are model names, values are predicted labels arrays
        - y_pred_probas: dict or None, keys are model names, values are predicted probabilities arrays
        - class_names: list of str or None, class names for reports and plots
        """
        self.model_names = model_names
        self.y_true = y_true
        self.y_preds = y_preds
        self.y_pred_probas = y_pred_probas
        self.class_names = class_names

    def generate_classification_reports(self):
        """
        Generate classification reports for all models.

        Returns:
        - reports: dict, keys are model names, values are classification report strings
        """
        reports = {}
        for model in self.model_names:
            report = classification_report(self.y_true, self.y_preds[model], target_names=self.class_names, zero_division=0)
            reports[model] = report
        return reports

    def plot_confusion_matrices(self, normalize=True, figsize=(15, 5)):
        """
        Plot confusion matrices for all models side by side.

        Parameters:
        - normalize: bool, whether to normalize confusion matrices
        - figsize: tuple, figure size
        """
        n_models = len(self.model_names)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        for ax, model in zip(axes, self.model_names):
            cm = confusion_matrix(self.y_true, self.y_preds[model])
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            ax.set_title(f'Confusion Matrix: {model}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def compare_accuracy(self):
        """
        Compute and return accuracy scores for all models.

        Returns:
        - accuracies: dict, keys are model names, values are accuracy floats
        """
        accuracies = {}
        for model in self.model_names:
            correct = np.sum(np.array(self.y_preds[model]) == np.array(self.y_true))
            accuracies[model] = correct / len(self.y_true)
        return accuracies

    def summary(self):
        """
        Print a summary of comparison including accuracy and classification reports.
        """
        accuracies = self.compare_accuracy()
        print("Model Accuracy Comparison:")
        for model, acc in accuracies.items():
            print(f" - {model}: {acc:.4f}")
        print("\nClassification Reports:")
        reports = self.generate_classification_reports()
        for model, report in reports.items():
            print(f"\nModel: {model}\n{report}")

# Additional utility functions can be added here as needed
