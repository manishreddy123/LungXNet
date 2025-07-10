"""
Image augmentation utilities for improved data processing and model performance.
"""

import tensorflow as tf

class ImageAugmentation:
    def __init__(self, config=None):
        self.config = config or {}
        self.augment_prob = self.config.get('augment_prob', 0.5)

    def random_flip(self, image):
        return tf.image.random_flip_left_right(image)

    def random_rotation(self, image):
        max_angle = self.config.get('max_rotation_angle', 0.1)  # radians
        angle = tf.random.uniform([], -max_angle, max_angle)
        # Use tf.image.experimental.rotate if available
        try:
            rotated_image = tf.image.experimental.rotate(image, angle)
        except AttributeError:
            # Fallback to no rotation if tf.image.experimental.rotate is not available
            rotated_image = image
        return rotated_image

    def random_zoom(self, image):
        scales = self.config.get('zoom_scales', [0.9, 1.1])
        scale = tf.random.uniform([], scales[0], scales[1])
        size = tf.shape(image)[:2]
        new_size = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
        return image

    def augment(self, image):
        if tf.random.uniform([]) > self.augment_prob:
            return image
        image = self.random_flip(image)
        image = self.random_rotation(image)
        image = self.random_zoom(image)
        return image
