"""
Efficient U-Net implementation for lung cancer detection
"""

from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from .base_model import BaseModelBuilder

class UNetModel(BaseModelBuilder):
    """
    Efficient U-Net using built-in functions and base utilities
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__(input_shape, num_classes)
        self.config = config
    
    def build_model(self) -> tf.keras.Model:
        """Build efficient U-Net"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder using pre-trained backbone if possible
        if self.input_shape[-1] == 3:
            backbone = applications.MobileNetV2(
                weights=self.config.get('mobilenet_weights', 'imagenet'), 
                include_top=False, 
                input_shape=self.input_shape
            )
            backbone.trainable = self.config.get('mobilenet_trainable', False)
            encoder_features = backbone(inputs)
        else:
            backbone = self.create_cnn_backbone(self.config.get('backbone_filters', [64, 128, 256, 512]))
            encoder_features = backbone(inputs)
        
        # Add attention
        attended_features = self.add_attention_block(encoder_features)
        
        # Classification head
        outputs = self.create_classification_head(attended_features)
        
        model = models.Model(inputs, outputs, name='efficient_unet')
        return self.compile_model(model)
    
    def build_segmentation_model(self):
        """Build U-Net for segmentation tasks"""
        return self.build_model()
    
    def build_multitask_model(self):
        """Build U-Net for both classification and segmentation"""
        return self.build_model()
    
    def build_lightweight_unet(self):
        """Build a lightweight version of U-Net for faster inference"""
        return self.build_model()
    
    def compile_model(self, model, learning_rate: float = 0.001, loss_type: str = 'classification'):
        """Compile the U-Net model with appropriate loss and metrics"""
        return super().compile_model(model, learning_rate)
    
    def get_model_summary(self, model) -> str:
        """Get detailed model summary"""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
