"""
Advanced Visualization utilities for model analysis and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import logging

class ModelVisualizer:
    """Comprehensive visualization toolkit for model analysis"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_dataset_distribution(self, dataset_info: Dict[str, Dict[str, int]], 
                                 save_path: Optional[str] = None):
        """Plot dataset distribution across splits and classes"""
        
        # Prepare data for plotting
        splits = list(dataset_info.keys())
        classes = self.class_names
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot for each split
        x = np.arange(len(classes))
        width = 0.25
        
        for i, split in enumerate(splits):
            counts = [dataset_info[split].get(cls, 0) for cls in classes]
            ax1.bar(x + i * width, counts, width, label=split.capitalize())
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Dataset Distribution by Class and Split')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pie chart for total distribution
        total_counts = {}
        for cls in classes:
            total_counts[cls] = sum(dataset_info[split].get(cls, 0) for split in splits)
        
        ax2.pie(total_counts.values(), labels=total_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Overall Class Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_images(self, images: np.ndarray, labels: np.ndarray, 
                          predictions: Optional[np.ndarray] = None,
                          num_samples: int = 12, save_path: Optional[str] = None):
        """Plot sample images with labels and predictions"""
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(min(num_samples, len(images))):
            ax = axes[i]
            
            # Display image
            if images[i].shape[-1] == 1:
                ax.imshow(images[i].squeeze(), cmap='gray')
            else:
                ax.imshow(images[i])
            
            # Create title
            true_label = self.class_names[labels[i]]
            title = f"True: {true_label}"
            
            if predictions is not None:
                pred_label = self.class_names[np.argmax(predictions[i])]
                confidence = np.max(predictions[i]) * 100
                title += f"\nPred: {pred_label} ({confidence:.1f}%)"
                
                # Color code based on correctness
                color = 'green' if labels[i] == np.argmax(predictions[i]) else 'red'
                ax.set_title(title, color=color, fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: Optional[str] = None):
        """Plot comprehensive training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Validation accuracy vs loss
        axes[1, 1].scatter(history['val_loss'], history['val_accuracy'], alpha=0.6)
        axes[1, 1].set_title('Validation Accuracy vs Loss')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = True, save_path: Optional[str] = None):
        """Plot enhanced confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       save_path: Optional[str] = None):
        """Plot ROC curves for multi-class classification"""
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(y_true, self.num_classes)
        else:
            y_true_onehot = y_true
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-Class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    save_path: Optional[str] = None):
        """Plot Precision-Recall curves for multi-class classification"""
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(y_true, self.num_classes)
        else:
            y_true_onehot = y_true
        
        plt.figure(figsize=(12, 8))
        
        # Plot PR curve for each class
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{self.class_names[i]} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-Class Classification')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_maps(self, model: tf.keras.Model, image: np.ndarray,
                         layer_names: List[str], save_path: Optional[str] = None):
        """Visualize feature maps from specified layers"""
        
        # Create a model that outputs feature maps
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(np.expand_dims(image, 0))
        
        # Plot feature maps
        fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, len(layer_names) * 3))
        
        for layer_idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
            for i in range(min(8, activation.shape[-1])):
                ax = axes[layer_idx, i] if len(layer_names) > 1 else axes[i]
                ax.imshow(activation[0, :, :, i], cmap='viridis')
                ax.set_title(f'{layer_name}\nFilter {i}')
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """Create interactive dashboard with plotly"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training History', 'Class Distribution', 
                          'Model Comparison', 'Prediction Confidence'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Training history
        if 'history' in results:
            history = results['history']
            epochs = list(range(len(history['accuracy'])))
            
            fig.add_trace(
                go.Scatter(x=epochs, y=history['accuracy'], name='Train Acc'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc'),
                row=1, col=1
            )
        
        # Class distribution
        if 'class_distribution' in results:
            fig.add_trace(
                go.Pie(labels=self.class_names, values=results['class_distribution']),
                row=1, col=2
            )
        
        # Model comparison
        if 'model_scores' in results:
            models = list(results['model_scores'].keys())
            scores = list(results['model_scores'].values())
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name='Model Accuracy'),
                row=2, col=1
            )
        
        # Prediction confidence
        if 'confidences' in results:
            fig.add_trace(
                go.Histogram(x=results['confidences'], name='Confidence Distribution'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Model Analysis Dashboard")
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None):
        """Compare multiple models across different metrics"""
        
        metrics = list(next(iter(model_results.values())).keys())
        models = list(model_results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_plots(self, results: Dict[str, Any], output_dir: str = 'plots'):
        """Save all visualization plots to specified directory"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Training history
            if 'history' in results:
                self.plot_training_history(
                    results['history'], 
                    save_path=os.path.join(output_dir, 'training_history.png')
                )
            
            # Confusion matrix
            if 'y_true' in results and 'y_pred' in results:
                self.plot_confusion_matrix(
                    results['y_true'], results['y_pred'],
                    save_path=os.path.join(output_dir, 'confusion_matrix.png')
                )
            
            # ROC curves
            if 'y_true' in results and 'y_pred_proba' in results:
                self.plot_roc_curves(
                    results['y_true'], results['y_pred_proba'],
                    save_path=os.path.join(output_dir, 'roc_curves.png')
                )
            
            # PR curves
            if 'y_true' in results and 'y_pred_proba' in results:
                self.plot_precision_recall_curves(
                    results['y_true'], results['y_pred_proba'],
                    save_path=os.path.join(output_dir, 'pr_curves.png')
                )
            
            # Model comparison
            if 'model_results' in results:
                self.plot_model_comparison(
                    results['model_results'],
                    save_path=os.path.join(output_dir, 'model_comparison.png')
                )
            
            self.logger.info(f"All plots saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving plots: {e}")
