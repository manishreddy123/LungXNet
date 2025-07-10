"""
Base model utilities for efficient CNN creation
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from typing import Tuple, List, Optional

class BaseModelBuilder:
    """Base class with common CNN building utilities"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_cnn_backbone(self, filters: List[int] = [64, 128, 256, 512], 
                           use_pretrained: bool = False) -> tf.keras.Model:
        """Create efficient CNN backbone using built-in functions"""
        
        if use_pretrained and self.input_shape[-1] == 3:
            # Use pre-trained backbone for RGB images
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            base_model.trainable = False  # Freeze initially
            return base_model
        
        # Create custom CNN backbone
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Efficient CNN blocks
        for i, f in enumerate(filters):
            x = self._conv_block(x, f, use_residual=(i > 0))
            if i < len(filters) - 1:
                x = layers.MaxPooling2D(2)(x)
        
        return models.Model(inputs, x, name='cnn_backbone')
    
    def _conv_block(self, inputs, filters: int, use_residual: bool = False):
        """Efficient convolutional block with residual connections"""
        
        x = layers.Conv2D(filters, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Residual connection
        if use_residual and inputs.shape[-1] == filters:
            x = layers.Add()([inputs, x])
        elif use_residual:
            shortcut = layers.Conv2D(filters, 1, padding='same')(inputs)
            shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([shortcut, x])
        
        x = layers.Activation('relu')(x)
        return x
    
    def create_classification_head(self, features, dropout_rate: float = 0.5):
        """Create efficient classification head"""
        
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return outputs
    
    def add_attention_block(self, inputs, ratio: int = 16):
        """Add efficient attention mechanism"""
        
        # Channel attention
        channels = inputs.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
        avg_pool = layers.Dense(channels // ratio, activation='relu')(avg_pool)
        avg_pool = layers.Dense(channels, activation='sigmoid')(avg_pool)
        
        # Apply attention
        attended = layers.Multiply()([inputs, avg_pool])
        
        return attended
    
    def compile_model(self, model: tf.keras.Model, learning_rate: float = 0.001):
        """Standard model compilation"""
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
