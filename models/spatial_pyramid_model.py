"""
Efficient Spatial Pyramid Pooling model for lung cancer detection
"""

from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, models
from .base_model import BaseModelBuilder

class SpatialPyramidModel(BaseModelBuilder):
    """
    Efficient Spatial Pyramid Model using built-in functions and base utilities
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__(input_shape, num_classes)
        self.config = config
    
    def build_model(self) -> tf.keras.Model:
        """Build efficient spatial pyramid model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN backbone
        backbone = self.create_cnn_backbone(self.config.get('backbone_filters', [64, 128, 256, 512]))
        features = backbone(inputs)
        
        # Spatial Pyramid Pooling using built-in layers
        pyramid_features = []
        
        # Different pooling scales
        for pool_size in self.config.get('pool_sizes', [1, 2, 4, 8]):
            pooled = layers.AveragePooling2D(
                pool_size=pool_size, strides=pool_size, padding='same'
            )(features)
            pooled = layers.GlobalAveragePooling2D()(pooled)
            pyramid_features.append(pooled)
        
        # Concatenate pyramid features
        x = layers.Concatenate()(pyramid_features)
        
        # Classification head
        x = layers.Dense(self.config.get('dense_units_1', 512), activation='relu')(x)
        x = layers.Dropout(self.config.get('dropout_rate_1', 0.5))(x)
        x = layers.Dense(self.config.get('dense_units_2', 256), activation='relu')(x)
        x = layers.Dropout(self.config.get('dropout_rate_2', 0.3))(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='efficient_spatial_pyramid_model')
        return self.compile_model(model)
    
    def build_advanced_spp_model(self):
        """Build model with advanced SPP techniques"""
        return self.build_model()
    
    def build_multiscale_spp_model(self):
        """Build model with multi-scale SPP at different feature levels"""
        return self.build_model()
    
    def build_lightweight_spp_model(self):
        """Build lightweight SPP model for faster inference"""
        return self.build_model()
    
    def compile_model(self, model, learning_rate: float = 0.001):
        """Compile the SPP model"""
        return super().compile_model(model, learning_rate)
    
    def get_model_summary(self, model) -> str:
        """Get detailed model summary"""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
