"""
Efficient Hybrid model for advanced lung cancer detection
"""

from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, models
from .base_model import BaseModelBuilder

class HybridUNetCapsule(BaseModelBuilder):
    """
    Efficient Hybrid Model using built-in functions and base utilities
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__(input_shape, num_classes)
        self.config = config
    
    def build_model(self) -> tf.keras.Model:
        """Build efficient hybrid model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Shared backbone
        backbone = self.create_cnn_backbone(self.config.get('backbone_filters', [64, 128, 256, 512]))
        shared_features = backbone(inputs)
        
        # Branch 1: Attention features
        attention_features = self.add_attention_block(shared_features)
        attention_pooled = layers.GlobalAveragePooling2D()(attention_features)
        
        # Branch 2: Spatial pyramid features
        pyramid_pooled = []
        for pool_size in self.config.get('pool_sizes', [1, 2, 4]):
            pooled = layers.AveragePooling2D(pool_size, padding='same')(shared_features)
            pooled = layers.GlobalAveragePooling2D()(pooled)
            pyramid_pooled.append(pooled)
        pyramid_features = layers.Concatenate()(pyramid_pooled)
        
        # Combine branches
        combined = layers.Concatenate()([attention_pooled, pyramid_features])
        
        # Final classification
        x = layers.Dense(self.config.get('dense_units', 512), activation='relu')(combined)
        x = layers.Dropout(self.config.get('dropout_rate', 0.5))(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='efficient_hybrid_model')
        return self.compile_model(model)
    
    def build_multibranch_model(self):
        """Build model with multiple output branches"""
        return self.build_model()
    
    def compile_model(self, model, learning_rate: float = 0.001):
        """Compile the hybrid model"""
        return super().compile_model(model, learning_rate)
    
    def get_model_summary(self, model) -> str:
        """Get detailed model summary"""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
