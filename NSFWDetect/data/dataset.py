"""
Dataset handler for NSFWDetect.
"""
import tensorflow as tf
import os
import random
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging

from config import (
    DATA_DIR, BATCH_SIZE, IMAGE_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    CATEGORY_MAPPING, NSFWCategory
)
from data.preprocessing import NSFWImagePreprocessor, BatchAugmentations

class NSFWDataset:
    """
    Dataset manager for NSFW image classification.

    Features:
    - Efficient data loading with TFRecord support
    - Advanced augmentation with mixup and cutmix
    - Caching and prefetching for performance
    - Class balancing and weighted sampling
    """
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        image_size: int = IMAGE_SIZE,
        use_cache: bool = True,
        balanced_sampling: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_cache = use_cache
        self.balanced_sampling = balanced_sampling

        # Initialize preprocessor
        self.train_preprocessor = NSFWImagePreprocessor(
            image_size=image_size, augment=True, normalize=True
        )
        self.eval_preprocessor = NSFWImagePreprocessor(
            image_size=image_size, augment=False, normalize=True
        )

        # Initialize batch augmentation
        self.batch_augmentations = BatchAugmentations()

        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found.")

        # Class mapping
        self.class_names = list(CATEGORY_MAPPING.values())
        self.num_classes = len(self.class_names)

        # Set up class weights for balanced training
        self.class_weights = self._compute_class_weights()

        # Set up dataset splits
        self.train_files, self.val_files, self.test_files = self._split_dataset()

        logging.info(f"Found {len(self.train_files)} training files, "
                     f"{len(self.val_files)} validation files, "
                     f"{len(self.test_files)} test files.")

    def _compute_class_weights(self) -> Dict[int, float]:
        """
        Compute class weights based on the distribution of samples in the dataset.
        Returns a dictionary mapping class indices to weights.
        """
        # Count samples per class
        class_counts = {class_name: 0 for class_name in self.class_names}

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                class_counts[class_name] = len(list(class_dir.glob('*.jpg'))) + \
                                          len(list(class_dir.glob('*.png'))) + \
                                          len(list(class_dir.glob('*.jpeg')))

        # Calculate weights inversely proportional to class frequencies
        total_samples = sum(class_counts.values())
        class_weights = {}

        for i, class_name in enumerate(self.class_names):
            if class_counts[class_name] > 0:
                class_weights[i] = total_samples / (len(self.class_names) * class_counts[class_name])
            else:
                class_weights[i] = 1.0

        # Normalize weights
        weight_sum = sum(class_weights.values())
        for i in class_weights:
            class_weights[i] = class_weights[i] / weight_sum * len(class_weights)

        return class_weights

    def _split_dataset(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split the dataset into training, validation, and test sets.
        """
        all_files = []

        # Collect all image files along with their class labels
        for i, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logging.warning(f"Class directory {class_dir} not found.")
                continue

            # Get all image files
            class_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))

            # Add class label to each file
            labeled_files = [(file, i) for file in class_files]
            all_files.extend(labeled_files)

        # Shuffle files
        random.shuffle(all_files)

        # Calculate split indices
        train_end = int(len(all_files) * TRAIN_SPLIT)
        val_end = train_end + int(len(all_files) * VAL_SPLIT)

        # Split files
        train_files = [f[0] for f in all_files[:train_end]]
        val_files = [f[0] for f in all_files[train_end:val_end]]
        test_files = [f[0] for f in all_files[val_end:]]

        return train_files, val_files, test_files

    def _parse_image(self, file_path: str) -> Tuple[tf.Tensor, int]:
        """
        Parse an image file and its label.
        """
        # Extract class name from path
        parts = tf.strings.split(file_path, os.path.sep)
        class_name = parts[-2]

        # Get class index
        class_index = tf.constant(0, dtype=tf.int32)
        for i, name in enumerate(self.class_names):
            class_index = tf.where(
                tf.equal(class_name, name),
                tf.constant(i, dtype=tf.int32),
                class_index
            )

        # Read and decode image
        img_bytes = tf.io.read_file(file_path)
        img = tf.image.decode_image(
            img_bytes, channels=3, expand_animations=False
        )
        img.set_shape([None, None, 3])

        return img, class_index

    def _prepare_sample(
        self, image: tf.Tensor, label: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prepare a single sample by applying preprocessing and converting label to one-hot.
        """
        # Apply preprocessing
        preprocessor = self.train_preprocessor if training else self.eval_preprocessor
        image = preprocessor.preprocess(image, training=training)

        # Convert label to one-hot
        label = tf.one_hot(label, self.num_classes)

        return image, label

    def _apply_batch_augmentations(
        self, images: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply batch-level augmentations."""
        return self.batch_augmentations.smart_augment(images, labels)

    def _create_dataset(
        self, file_paths: List[Path], training: bool = False
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from a list of file paths.
        """
        # Convert file paths to strings
        file_paths = [str(file) for file in file_paths]

        # Create dataset from file paths
        ds = tf.data.Dataset.from_tensor_slices(file_paths)

        # Shuffle if training
        if training:
            ds = ds.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

        # Parse images and labels
        ds = ds.map(
            self._parse_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply preprocessing and convert labels
        ds = ds.map(
            lambda x, y: self._prepare_sample(x, y, training=training),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Cache if requested
        if self.use_cache:
            ds = ds.cache()

        # Batch
        ds = ds.batch(self.batch_size)

        # Apply batch augmentations during training
        if training:
            ds = ds.map(
                self._apply_batch_augmentations,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Prefetch
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def get_train_dataset(self) -> tf.data.Dataset:
        """Get the training dataset."""
        return self._create_dataset(self.train_files, training=True)

    def get_val_dataset(self) -> tf.data.Dataset:
        """Get the validation dataset."""
        return self._create_dataset(self.val_files, training=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        """Get the test dataset."""
        return self._create_dataset(self.test_files, training=False)

    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for balanced training."""
        return self.class_weights

    def get_steps_per_epoch(self) -> int:
        """Get the number of steps per epoch."""
        return len(self.train_files) // self.batch_size + 1

    def get_validation_steps(self) -> int:
        """Get the number of validation steps."""
        return len(self.val_files) // self.batch_size + 1
