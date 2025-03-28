"""
Training module for NSFWDetect.
"""
import tensorflow as tf
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, CHECKPOINT_DIR, TENSORBOARD_DIR
)
from models.vit_model import ViTModel
from data.dataset import NSFWDataset
from utils.tensorboard import NSFWTensorBoard

class NSFWTrainer:
    """
    Trainer class for NSFW image classification using Vision Transformer.

    Features:
    - Mixed precision training for faster performance
    - LR scheduling with warmup
    - Gradient accumulation
    - Advanced metrics tracking
    - Best model checkpointing
    """
    def __init__(
        self,
        model: ViTModel,
        dataset: NSFWDataset,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_steps: int = WARMUP_STEPS,
        checkpoint_dir: str = CHECKPOINT_DIR,
        tensorboard_dir: str = TENSORBOARD_DIR,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.dataset = dataset
        self.initial_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Set up mixed precision
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Using mixed precision training.")

        # Set up optimizer with LR schedule
        self.optimizer = self._create_optimizer()

        # Set up metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.train_precision = tf.keras.metrics.Precision(name='train_precision')
        self.train_recall = tf.keras.metrics.Recall(name='train_recall')
        self.train_f1 = tf.keras.metrics.F1Score(name='train_f1', average='weighted')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        self.val_precision = tf.keras.metrics.Precision(name='val_precision')
        self.val_recall = tf.keras.metrics.Recall(name='val_recall')
        self.val_f1 = tf.keras.metrics.F1Score(name='val_f1', average='weighted')

        # Set up TensorBoard
        self.tensorboard = NSFWTensorBoard(
            log_dir=str(self.tensorboard_dir),
            histogram_freq=1,
            update_freq='epoch'
        )

        # Set class names for TensorBoard
        class_names = [dataset.class_names[i] for i in range(len(dataset.class_names))]
        self.tensorboard.set_class_names(class_names)

        # Set up loss function with class weights
        self.class_weights = self.dataset.get_class_weights()
        self.loss_fn = self._weighted_categorical_crossentropy()

        # Set up checkpointing
        self.checkpoint_manager = self._setup_checkpointing()

        # Track best model
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer with learning rate schedule."""
        # Learning rate schedule with warmup
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=self.dataset.get_steps_per_epoch() * NUM_EPOCHS - self.warmup_steps,
            end_learning_rate=0.0,
            power=1.0
        )

        # Add warmup to the schedule
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=self.warmup_steps,
            end_learning_rate=self.initial_learning_rate,
            power=1.0
        )

        # Combine warmup and decay schedules
        def lr_fn(step):
            return tf.cond(
                step < self.warmup_steps,
                lambda: warmup_schedule(step),
                lambda: lr_schedule(step - self.warmup_steps)
            )

        # Create AdamW optimizer (using Keras implementation instead of tensorflow_addons)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_fn,
            weight_decay=self.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        # If using mixed precision, wrap optimizer in LossScaleOptimizer
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        return optimizer

    def _weighted_categorical_crossentropy(self):
        """Create weighted categorical crossentropy loss function."""
        class_weights_tensor = tf.constant(
            [self.class_weights[i] for i in range(len(self.class_weights))],
            dtype=tf.float32
        )

        def loss_fn(y_true, y_pred):
            # Convert one-hot to indices for weight lookup
            y_true_indices = tf.argmax(y_true, axis=1)
            sample_weights = tf.gather(class_weights_tensor, y_true_indices)

            # Apply crossentropy with sample weights
            cce = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss = cce(y_true, y_pred)
            weighted_loss = loss * sample_weights

            return tf.reduce_mean(weighted_loss)

        return loss_fn

    def _setup_checkpointing(self) -> tf.train.CheckpointManager:
        """Set up model checkpointing."""
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        return tf.train.CheckpointManager(
            checkpoint,
            str(self.checkpoint_dir),
            max_to_keep=5
        )

    @tf.function
    def _train_step(
        self, images: tf.Tensor, labels: tf.Tensor, step: int, accumulation_step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Execute a single training step.

        Args:
            images: Batch of images
            labels: Batch of labels
            step: Global training step
            accumulation_step: Current accumulation step

        Returns:
            Tuple of (loss, predictions)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(images, training=True)

            # Calculate loss
            loss = self.loss_fn(labels, predictions)

            # Scale loss if using mixed precision
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

            # Scale loss by gradient accumulation steps
            scaled_loss = scaled_loss / tf.cast(self.gradient_accumulation_steps, scaled_loss.dtype)

        # Calculate gradients
        if self.use_mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Only apply gradients at end of accumulation or last batch
        if (accumulation_step == self.gradient_accumulation_steps - 1) or (step == 0):
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, predictions

    @tf.function
    def _val_step(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Execute a single validation step."""
        predictions = self.model(images, training=False)
        loss = self.loss_fn(labels, predictions)
        return loss, predictions

    def train(self, epochs: int = NUM_EPOCHS) -> Dict[str, list]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train for

        Returns:
            Dictionary of training history
        """
        # Get datasets
        train_ds = self.dataset.get_train_dataset()
        val_ds = self.dataset.get_val_dataset()

        # Get TensorBoard callback for the model
        tensorboard_callback = self.tensorboard.get_callback(self.model)

        # Initialize training history
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_f1': []
        }

        # Training loop
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            start_time = time.time()

            # Reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_precision.reset_states()
            self.train_recall.reset_states()
            self.train_f1.reset_states()

            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            self.val_precision.reset_states()
            self.val_recall.reset_states()
            self.val_f1.reset_states()

            # Training phase
            for step, (images, labels) in enumerate(train_ds):
                # Determine accumulation step
                accumulation_step = step % self.gradient_accumulation_steps

                # Execute training step
                loss, predictions = self._train_step(images, labels, step, accumulation_step)

                # Update metrics
                self.train_loss.update_state(loss)
                self.train_accuracy.update_state(labels, predictions)
                self.train_precision.update_state(labels, tf.nn.softmax(predictions))
                self.train_recall.update_state(labels, tf.nn.softmax(predictions))
                self.train_f1.update_state(labels, tf.nn.softmax(predictions))

                # Log progress
                if step % 50 == 0:
                    logging.info(
                        f"Step {step}, "
                        f"Loss: {self.train_loss.result():.4f}, "
                        f"Accuracy: {self.train_accuracy.result():.4f}, "
                        f"F1: {self.train_f1.result():.4f}"
                    )

            # Validation phase
            all_val_labels = []
            all_val_predictions = []
            sample_images = None
            sample_labels = None
            sample_preds = None

            for i, (images, labels) in enumerate(val_ds):
                loss, predictions = self._val_step(images, labels)

                # Update metrics
                self.val_loss.update_state(loss)
                self.val_accuracy.update_state(labels, predictions)
                self.val_precision.update_state(labels, tf.nn.softmax(predictions))
                self.val_recall.update_state(labels, tf.nn.softmax(predictions))
                self.val_f1.update_state(labels, tf.nn.softmax(predictions))

                # Save predictions and labels for confusion matrix
                all_val_labels.append(tf.argmax(labels, axis=1).numpy())
                all_val_predictions.append(tf.argmax(tf.nn.softmax(predictions), axis=1).numpy())

                # Save first batch for prediction visualization
                if i == 0:
                    sample_images = images.numpy()
                    sample_labels = tf.argmax(labels, axis=1).numpy()
                    sample_preds = tf.argmax(tf.nn.softmax(predictions), axis=1).numpy()
                    sample_probs = tf.nn.softmax(predictions).numpy()

            # Concatenate all validation predictions and labels
            all_val_labels = np.concatenate(all_val_labels)
            all_val_predictions = np.concatenate(all_val_predictions)

            # Log confusion matrix and sample predictions to TensorBoard
            self.tensorboard.log_confusion_matrix(all_val_labels, all_val_predictions, epoch)
            if sample_images is not None:
                self.tensorboard.log_sample_predictions(
                    sample_images, sample_labels, sample_preds, sample_probs, epoch
                )

            # Log learning rate to TensorBoard
            self.tensorboard.log_model_weights_histogram(self.model, epoch)
            self.tensorboard.log_learning_rate(self.optimizer, epoch)

            # Calculate epoch metrics
            train_loss = self.train_loss.result()
            train_accuracy = self.train_accuracy.result()
            train_precision = self.train_precision.result()
            train_recall = self.train_recall.result()
            train_f1 = self.train_f1.result()

            val_loss = self.val_loss.result()
            val_accuracy = self.val_accuracy.result()
            val_precision = self.val_precision.result()
            val_recall = self.val_recall.result()
            val_f1 = self.val_f1.result()

            # Update history
            history['train_loss'].append(float(train_loss))
            history['train_accuracy'].append(float(train_accuracy))
            history['train_f1'].append(float(train_f1))
            history['val_loss'].append(float(val_loss))
            history['val_accuracy'].append(float(val_accuracy))
            history['val_f1'].append(float(val_f1))

            # Log epoch results
            epoch_time = time.time() - start_time
            logging.info(
                f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                f"F1: {train_f1:.4f}, Precision: {train_precision:.4f}, "
                f"Recall: {train_recall:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val Recall: {val_recall:.4f}"
            )

            # Write metrics to TensorBoard
            with self.tensorboard.train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', train_accuracy, step=epoch)
                tf.summary.scalar('precision', train_precision, step=epoch)
                tf.summary.scalar('recall', train_recall, step=epoch)
                tf.summary.scalar('f1', train_f1, step=epoch)

            with self.tensorboard.val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step=epoch)
                tf.summary.scalar('accuracy', val_accuracy, step=epoch)
                tf.summary.scalar('precision', val_precision, step=epoch)
                tf.summary.scalar('recall', val_recall, step=epoch)
                tf.summary.scalar('f1', val_f1, step=epoch)

            # Save checkpoint if best model
            if val_accuracy > self.best_val_accuracy:
                logging.info(f"Validation accuracy improved from {self.best_val_accuracy:.4f} to {val_accuracy:.4f}")
                self.best_val_accuracy = val_accuracy
                self.checkpoint_manager.save(checkpoint_number=epoch)

                # If accuracy is over 95%, save a special checkpoint
                if val_accuracy >= 0.95:
                    special_checkpoint_path = self.checkpoint_dir / f"accuracy_95plus_epoch_{epoch}"
                    self.checkpoint_manager.save(checkpoint_number=1000 + epoch)
                    logging.info(f"Achieved 95%+ accuracy! Saved checkpoint at {special_checkpoint_path}")

            # Early stopping check based on F1 score (more appropriate for imbalanced data)
            if val_f1 > self.best_val_f1:
                logging.info(f"Validation F1 improved from {self.best_val_f1:.4f} to {val_f1:.4f}")
                self.best_val_f1 = val_f1
                # No need to save again if we already saved for accuracy

        logging.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
        logging.info(f"Best validation F1 score: {self.best_val_f1:.4f}")

        return history
