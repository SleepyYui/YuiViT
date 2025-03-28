"""
Data preprocessing utilities for NSFWDetect.
"""
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from config import IMAGE_SIZE

class NSFWImagePreprocessor:
    """
    Advanced preprocessing for NSFW image classification using 2025 techniques.
    """
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        augment: bool = True,
        normalize: bool = True
    ):
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

        # Constants for normalization
        self.mean = tf.constant([0.485, 0.456, 0.406])
        self.std = tf.constant([0.229, 0.224, 0.225])

        # Augmentation parameters
        self.aug_params = {
            'rotation_range': 15,
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'noise_factor': 0.05,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 0.2
        }

    def resize_with_aspect_ratio(self, image: tf.Tensor) -> tf.Tensor:
        """Resize image while preserving aspect ratio."""
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        # Calculate scale factor
        scale = tf.maximum(
            self.image_size / tf.cast(height, tf.float32),
            self.image_size / tf.cast(width, tf.float32)
        )

        # New dimensions
        new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

        # Resize
        image = tf.image.resize(image, [new_height, new_width], preserve_aspect_ratio=True)

        # Central crop
        image = tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)

        return image

    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """Apply normalization."""
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - self.mean) / self.std
        return image

    def advanced_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply advanced augmentation techniques."""
        # Random rotation
        if tf.random.uniform([]) < 0.5:
            degrees = self.aug_params['rotation_range'] * tf.random.uniform([], -1, 1)
            image = tfa.image.rotate(image, degrees * np.pi / 180)

        # Random brightness and contrast
        if tf.random.uniform([]) < 0.5:
            brightness_factor = tf.random.uniform(
                [],
                self.aug_params['brightness_range'][0],
                self.aug_params['brightness_range'][1]
            )
            image = tf.image.adjust_brightness(image, brightness_factor - 1.0)

        if tf.random.uniform([]) < 0.5:
            contrast_factor = tf.random.uniform(
                [],
                self.aug_params['contrast_range'][0],
                self.aug_params['contrast_range'][1]
            )
            image = tf.image.adjust_contrast(image, contrast_factor)

        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Random noise (simulating 2025 techniques for preserving structure)
        if tf.random.uniform([]) < 0.3:
            noise = tf.random.normal(
                tf.shape(image),
                mean=0.0,
                stddev=self.aug_params['noise_factor']
            )
            image = tf.clip_by_value(image + noise, 0.0, 1.0)

        # Random color jitter - advanced version for 2025
        if tf.random.uniform([]) < 0.3:
            # Randomize the order of color transforms
            transforms = tf.random.shuffle(tf.range(4))
            for transform in transforms:
                if transform == 0:
                    image = tf.image.random_hue(image, 0.05)
                elif transform == 1:
                    image = tf.image.random_saturation(image, 0.9, 1.1)
                elif transform == 2:
                    image = tf.image.random_brightness(image, 0.1)
                else:
                    image = tf.image.random_contrast(image, 0.9, 1.1)

        return image

    def preprocess(self, image: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Main preprocessing function."""
        # Ensure image is a tensor with float values
        image = tf.cast(image, tf.float32)

        # Resize with aspect ratio preservation
        image = self.resize_with_aspect_ratio(image)

        # Apply normalization before augmentation for stability
        if self.normalize:
            image = self.normalize_image(image)

        # Apply augmentations only during training
        if training and self.augment:
            image = self.advanced_augmentation(image)

        return image

    def build_preprocessing_pipeline(self, training: bool = False):
        """Return a function that applies the preprocessing pipeline."""
        def _preprocess(image):
            return self.preprocess(image, training=training)
        return _preprocess


# Advanced augmentation techniques that can be applied at the batch level
class BatchAugmentations:
    """
    Advanced batch-level augmentations for NSFW detection,
    implementing techniques that are expected to be state-of-the-art in 2025.
    """
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 0.2):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def mixup(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply mixup augmentation to a batch of images and labels.

        Args:
            images: Tensor of shape [batch_size, height, width, channels]
            labels: One-hot encoded tensor of shape [batch_size, num_classes]

        Returns:
            Tuple of (mixed images, mixed labels)
        """
        batch_size = tf.shape(images)[0]

        # Generate random indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))

        # Generate mixup ratio from beta distribution
        alpha = self.mixup_alpha
        lam = tf.random.uniform([], 0, 1) if alpha <= 0 else \
              tf.random.beta(alpha, alpha, [batch_size, 1, 1, 1])

        # Mix images
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)

        # Mix labels
        lam_flat = tf.reshape(lam, [batch_size, 1])
        mixed_labels = lam_flat * labels + (1 - lam_flat) * tf.gather(labels, indices)

        return mixed_images, mixed_labels

    def cutmix(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply cutmix augmentation to a batch of images and labels.

        Args:
            images: Tensor of shape [batch_size, height, width, channels]
            labels: One-hot encoded tensor of shape [batch_size, num_classes]

        Returns:
            Tuple of (cut-mixed images, cut-mixed labels)
        """
        batch_size = tf.shape(images)[0]
        img_height, img_width = tf.shape(images)[1], tf.shape(images)[2]

        # Generate random indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))

        # Generate cutmix ratio from beta distribution
        alpha = self.cutmix_alpha
        lam = tf.random.uniform([], 0, 1) if alpha <= 0 else \
              tf.random.beta(alpha, alpha, [])

        # Get random box coordinates
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_height = tf.cast(tf.cast(img_height, tf.float32) * cut_ratio, tf.int32)
        cut_width = tf.cast(tf.cast(img_width, tf.float32) * cut_ratio, tf.int32)

        # Calculate center of the box
        center_y = tf.random.uniform([], 0, img_height, dtype=tf.int32)
        center_x = tf.random.uniform([], 0, img_width, dtype=tf.int32)

        # Calculate box boundaries
        y1 = tf.maximum(0, center_y - cut_height // 2)
        y2 = tf.minimum(img_height, center_y + cut_height // 2)
        x1 = tf.maximum(0, center_x - cut_width // 2)
        x2 = tf.minimum(img_width, center_x + cut_width // 2)

        # Create mask for the box
        mask = tf.ones((batch_size, img_height, img_width, 1))
        box_mask = tf.zeros((y2 - y1, x2 - x1, 1))

        # Update mask with box
        indices_i = tf.reshape(tf.range(y1, y2), [-1, 1])
        indices_j = tf.reshape(tf.range(x1, x2), [1, -1])
        indices = tf.reshape(tf.stack([tf.repeat(indices_i, x2 - x1, axis=1),
                                        tf.repeat(indices_j, y2 - y1, axis=0)], axis=2), [-1, 2])

        updates = tf.ones((tf.shape(indices)[0], 1))
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        # Apply mask to create cutmix images
        mixed_images = images * mask + tf.gather(images, indices) * (1 - mask)

        # Calculate new mixing ratio based on area
        lam = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / \
              tf.cast(img_height * img_width, tf.float32)

        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

        return mixed_images, mixed_labels

    def smart_augment(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply either mixup or cutmix based on random selection.
        For NSFW detection, we bias towards mixup which often works better
        for preserving subtle signals in sensitive content.
        """
        # Randomly choose between mixup and cutmix, with bias towards mixup
        augmentation_choice = tf.random.uniform([], 0, 1)

        if augmentation_choice < 0.7:  # 70% chance of mixup
            return self.mixup(images, labels)
        else:
            return self.cutmix(images, labels)
