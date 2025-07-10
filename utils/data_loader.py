"""
Advanced Data Loading utilities for lung cancer detection
"""

import tensorflow as tf
import numpy as np
import os
from typing import Tuple, Dict, Any
import logging

class DataLoader:
    """Advanced data loader with comprehensive preprocessing and augmentation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.channels = config['channels']
        self.num_classes = config['num_classes']
        self.class_names = config['class_names']
        self.data_paths = config['data_paths']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_dataset_info(self, folder_path: str) -> Dict[str, int]:
        """Get information about dataset distribution"""
        dataset_info = {}
        try:
            for class_folder in os.listdir(folder_path):
                class_path = os.path.join(folder_path, class_folder)
                if os.path.isdir(class_path):
                    num_images = len([f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    dataset_info[class_folder] = num_images
        except Exception as e:
            self.logger.error(f"Error reading dataset info: {e}")
        return dataset_info
    
    def create_dataset(self, folder_path: str, is_training: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset"""
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                folder_path,
                shuffle=is_training,
                image_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                color_mode="grayscale" if self.channels == 1 else "rgb",
                label_mode='int'
            )
            
            # Optimize dataset performance
            dataset = dataset.cache()
            if is_training:
                dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            self.logger.info(f"Created dataset from {folder_path}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}")
            return None
    
    def load_all_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load train, validation, and test datasets"""
        train_ds = self.create_dataset(self.data_paths['train'], is_training=True)
        val_ds = self.create_dataset(self.data_paths['val'], is_training=False)
        test_ds = self.create_dataset(self.data_paths['test'], is_training=False)
        
        return train_ds, val_ds, test_ds
    
    def get_class_weights(self, dataset: tf.data.Dataset) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset"""
        try:
            # Count samples per class
            class_counts = np.zeros(self.num_classes)
            for _, labels in dataset:
                for label in labels:
                    class_counts[label.numpy()] += 1
            
            # Calculate weights (inverse frequency)
            total_samples = np.sum(class_counts)
            class_weights = {}
            for i in range(self.num_classes):
                if class_counts[i] > 0:
                    class_weights[i] = total_samples / (self.num_classes * class_counts[i])
                else:
                    class_weights[i] = 1.0
            
            self.logger.info(f"Class weights calculated: {class_weights}")
            return class_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating class weights: {e}")
            return {i: 1.0 for i in range(self.num_classes)}
    
    def preprocess_dataset(self, dataset: tf.data.Dataset, 
                          apply_augmentation: bool = False) -> tf.data.Dataset:
        """Apply preprocessing and optional augmentation"""
        
        # Normalization
        normalization = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1.0/255.0)
        ])
        
        if apply_augmentation:
            # Data augmentation for training
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomBrightness(0.2)
            ])
            
            preprocessing = tf.keras.Sequential([
                augmentation,
                normalization
            ])
        else:
            preprocessing = normalization
        
        # Apply preprocessing
        dataset = dataset.map(
            lambda x, y: (preprocessing(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset
    
    def create_stratified_split(self, dataset: tf.data.Dataset, 
                               split_ratio: float = 0.8) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create stratified train/validation split"""
        try:
            # Convert to numpy for stratification
            images, labels = [], []
            for batch_images, batch_labels in dataset:
                images.extend(batch_images.numpy())
                labels.extend(batch_labels.numpy())
            
            images = np.array(images)
            labels = np.array(labels)
            
            # Stratified split
            from sklearn.model_selection import train_test_split
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, test_size=1-split_ratio, 
                stratify=labels, random_state=42
            )
            
            # Convert back to tf.data.Dataset
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
            
            # Batch the datasets
            train_ds = train_ds.batch(self.batch_size)
            val_ds = val_ds.batch(self.batch_size)
            
            return train_ds, val_ds
            
        except Exception as e:
            self.logger.error(f"Error in stratified split: {e}")
            return dataset, dataset
    
    def get_sample_batch(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample batch for visualization"""
        for images, labels in dataset.take(1):
            return images.numpy(), labels.numpy()
        return None, None
    
    def print_dataset_info(self):
        """Print comprehensive dataset information"""
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        for split_name, path in self.data_paths.items():
            if os.path.exists(path):
                info = self.get_dataset_info(path)
                print(f"\n{split_name.upper()} SET:")
                total = sum(info.values())
                for class_name, count in info.items():
                    percentage = (count / total) * 100 if total > 0 else 0
                    print(f"  {class_name}: {count} images ({percentage:.1f}%)")
                print(f"  Total: {total} images")
            else:
                print(f"\n{split_name.upper()} SET: Path not found - {path}")
        
        print(f"\nConfiguration:")
        print(f"  Image Size: {self.image_size}x{self.image_size}")
        print(f"  Channels: {self.channels}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Classes: {self.num_classes}")
        print("=" * 60)
