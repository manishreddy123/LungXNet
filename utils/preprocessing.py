"""
Advanced Image Preprocessing utilities
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging

class ImagePreprocessor:
    """Advanced image preprocessing with medical image specific techniques"""
    
    def __init__(self, image_size: int = 256, channels: int = 1):
        self.image_size = image_size
        self.channels = channels
        self.logger = logging.getLogger(__name__)
    
    def normalize_image(self, image: tf.Tensor, method: str = 'standard') -> tf.Tensor:
        """Apply different normalization techniques"""
        if method == 'standard':
            # Standard normalization (0-1)
            return tf.cast(image, tf.float32) / 255.0
        elif method == 'z_score':
            # Z-score normalization
            mean = tf.reduce_mean(image)
            std = tf.math.reduce_std(image)
            return (tf.cast(image, tf.float32) - mean) / (std + 1e-8)
        elif method == 'min_max':
            # Min-Max normalization
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            return (tf.cast(image, tf.float32) - min_val) / (max_val - min_val + 1e-8)
        else:
            return tf.cast(image, tf.float32) / 255.0
    
    def apply_clahe(self, image: tf.Tensor, clip_limit: float = 2.0) -> tf.Tensor:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        def clahe_func(img):
            # Convert to numpy for OpenCV processing
            img_np = img.numpy().astype(np.uint8)
            if len(img_np.shape) == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_np)
            
            # Restore original shape
            if len(img.shape) == 3:
                enhanced = np.expand_dims(enhanced, -1)
            
            return enhanced.astype(np.float32)
        
        return tf.py_function(clahe_func, [image], tf.float32)
    
    def gaussian_noise(self, image: tf.Tensor, noise_factor: float = 0.1) -> tf.Tensor:
        """Add Gaussian noise for augmentation"""
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_factor)
        noisy_image = image + noise
        return tf.clip_by_value(noisy_image, 0.0, 1.0)
    
    def elastic_transform(self, image: tf.Tensor, alpha: float = 1.0, sigma: float = 50.0) -> tf.Tensor:
        """Apply elastic transformation for medical image augmentation"""
        def elastic_transform_func(img):
            img_np = img.numpy()
            shape = img_np.shape[:2]
            
            # Generate random displacement fields
            dx = np.random.uniform(-1, 1, shape) * alpha
            dy = np.random.uniform(-1, 1, shape) * alpha
            
            # Smooth the displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x_new = (x + dx).astype(np.float32)
            y_new = (y + dy).astype(np.float32)
            
            # Apply transformation
            if len(img_np.shape) == 3:
                transformed = cv2.remap(img_np, x_new, y_new, cv2.INTER_LINEAR)
            else:
                transformed = cv2.remap(img_np, x_new, y_new, cv2.INTER_LINEAR)
                if len(img.shape) == 3:
                    transformed = np.expand_dims(transformed, -1)
            
            return transformed.astype(np.float32)
        
        return tf.py_function(elastic_transform_func, [image], tf.float32)
    
    def morphological_operations(self, image: tf.Tensor, operation: str = 'opening') -> tf.Tensor:
        """Apply morphological operations"""
        def morph_func(img):
            img_np = (img.numpy() * 255).astype(np.uint8)
            if len(img_np.shape) == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            if operation == 'opening':
                result = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                result = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                result = cv2.morphologyEx(img_np, cv2.MORPH_GRADIENT, kernel)
            else:
                result = img_np
            
            if len(img.shape) == 3:
                result = np.expand_dims(result, -1)
            
            return (result / 255.0).astype(np.float32)
        
        return tf.py_function(morph_func, [image], tf.float32)
    
    def edge_enhancement(self, image: tf.Tensor, method: str = 'sobel') -> tf.Tensor:
        """Enhance edges in medical images"""
        def edge_func(img):
            img_np = img.numpy()
            if len(img_np.shape) == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            
            if method == 'sobel':
                grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(grad_x**2 + grad_y**2)
            elif method == 'laplacian':
                edges = cv2.Laplacian(img_np, cv2.CV_64F)
                edges = np.abs(edges)
            elif method == 'canny':
                edges = cv2.Canny((img_np * 255).astype(np.uint8), 50, 150)
                edges = edges / 255.0
            else:
                edges = img_np
            
            # Normalize edges
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
            
            if len(img.shape) == 3:
                edges = np.expand_dims(edges, -1)
            
            return edges.astype(np.float32)
        
        return tf.py_function(edge_func, [image], tf.float32)
    
    def create_preprocessing_pipeline(self, 
                                    apply_clahe: bool = True,
                                    apply_edge_enhancement: bool = False,
                                    normalization_method: str = 'standard') -> tf.keras.Sequential:
        """Create a comprehensive preprocessing pipeline"""
        
        layers = []
        
        # Resize if needed
        layers.append(tf.keras.layers.Resizing(self.image_size, self.image_size))
        
        # Apply CLAHE for contrast enhancement
        if apply_clahe:
            layers.append(tf.keras.layers.Lambda(
                lambda x: self.apply_clahe(x), name='clahe_enhancement'
            ))
        
        # Apply edge enhancement
        if apply_edge_enhancement:
            layers.append(tf.keras.layers.Lambda(
                lambda x: self.edge_enhancement(x), name='edge_enhancement'
            ))
        
        # Normalization
        layers.append(tf.keras.layers.Lambda(
            lambda x: self.normalize_image(x, normalization_method), 
            name=f'{normalization_method}_normalization'
        ))
        
        return tf.keras.Sequential(layers, name='preprocessing_pipeline')
    
    def create_advanced_augmentation(self) -> tf.keras.Sequential:
        """Create advanced augmentation pipeline for medical images"""
        
        return tf.keras.Sequential([
            # Geometric transformations
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15, fill_mode='reflect'),
            tf.keras.layers.RandomZoom(0.15, fill_mode='reflect'),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='reflect'),
            
            # Intensity transformations
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            
            # Custom augmentations
            tf.keras.layers.Lambda(
                lambda x: self.gaussian_noise(x, 0.05), 
                name='gaussian_noise'
            ),
            
        ], name='advanced_augmentation')
    
    def preprocess_batch(self, images: tf.Tensor, labels: tf.Tensor, 
                        training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess a batch of images"""
        
        # Apply preprocessing pipeline
        preprocessing = self.create_preprocessing_pipeline()
        processed_images = preprocessing(images)
        
        # Apply augmentation during training
        if training:
            augmentation = self.create_advanced_augmentation()
            processed_images = augmentation(processed_images)
        
        return processed_images, labels
    
    def visualize_preprocessing_steps(self, image: tf.Tensor) -> List[tf.Tensor]:
        """Visualize different preprocessing steps"""
        steps = []
        
        # Original
        steps.append(('Original', image))
        
        # CLAHE
        clahe_img = self.apply_clahe(image)
        steps.append(('CLAHE Enhanced', clahe_img))
        
        # Edge Enhancement
        edge_img = self.edge_enhancement(image)
        steps.append(('Edge Enhanced', edge_img))
        
        # Normalized
        norm_img = self.normalize_image(image)
        steps.append(('Normalized', norm_img))
        
        # With noise
        noisy_img = self.gaussian_noise(norm_img)
        steps.append(('With Noise', noisy_img))
        
        return steps
