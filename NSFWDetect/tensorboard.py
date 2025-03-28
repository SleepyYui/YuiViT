"""
TensorBoard utilities for NSFWDetect to visualize training and model architecture.
"""
import tensorflow as tf
import datetime
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

class NSFWTensorBoard:
    """
    Enhanced TensorBoard manager for NSFWDetect with additional visualizations.

    Features:
    - Model graph visualization
    - Training/validation metrics
    - Image predictions visualization
    - Confusion matrix logging
    - Embedding visualization
    - Distribution and histogram tracking
    """
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        histogram_freq: int = 1,
        update_freq: str = "epoch"
    ):
        """
        Initialize TensorBoard manager.

        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Optional name for this experiment (default: timestamped directory)
            histogram_freq: Frequency (in epochs) to compute histograms
            update_freq: Update frequency for TensorBoard ('batch' or 'epoch')
        """
        self.base_log_dir = Path(log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log directory
        if experiment_name:
            self.log_dir = self.base_log_dir / experiment_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = self.base_log_dir / f"run_{timestamp}"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create separate log directories for training and validation
        self.train_log_dir = self.log_dir / "train"
        self.val_log_dir = self.log_dir / "validation"
        self.train_log_dir.mkdir(parents=True, exist_ok=True)
        self.val_log_dir.mkdir(parents=True, exist_ok=True)

        # Create file writers
        self.train_summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))
        self.val_summary_writer = tf.summary.create_file_writer(str(self.val_log_dir))

        # Save histogram frequency
        self.histogram_freq = histogram_freq
        self.update_freq = update_freq

        # Store class names for visualization
        self.class_names = None

    def get_callback(self, model: Optional[tf.keras.Model] = None) -> tf.keras.callbacks.TensorBoard:
        """
        Get TensorBoard callback for Keras training.

        Args:
            model: Optional model to trace (for graph visualization)

        Returns:
            TensorBoard callback
        """
        callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=self.histogram_freq,
            write_graph=True,
            write_images=True,
            update_freq=self.update_freq,
            profile_batch='500,520'  # Profile a few batches for performance analysis
        )

        # If model is provided, trace it to visualize the graph
        if model is not None:
            # Write the graph to TensorBoard
            self._write_model_graph(model)

        return callback

    def _write_model_graph(self, model: tf.keras.Model) -> None:
        """
        Write model graph to TensorBoard.

        Args:
            model: Keras model to visualize
        """
        # Get input shape from model
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
            if isinstance(input_shape, tuple) and input_shape[0] is None:
                # Handle batch dimension
                input_shape = (1,) + input_shape[1:]
        else:
            # Default to a standard image shape if input_shape is not available
            input_shape = (1, 384, 384, 3)

        # Create a trace of the model execution
        @tf.function
        def trace_model(x):
            return model(x, training=False)

        # Create dummy input based on input shape
        dummy_input = tf.random.normal(input_shape)

        # Trace the model
        with self.train_summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            _ = trace_model(dummy_input)
            tf.summary.trace_export(
                name="model_trace",
                step=0,
                profiler_outdir=str(self.log_dir)
            )

    def set_class_names(self, class_names: List[str]) -> None:
        """
        Set class names for visualization.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names

    def log_confusion_matrix(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        step: int
    ) -> None:
        """
        Log confusion matrix to TensorBoard.

        Args:
            true_labels: Ground truth labels (indices)
            predicted_labels: Predicted labels (indices)
            step: Current step/epoch
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Normalize the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Use class names if available
        if self.class_names is not None:
            labels = self.class_names
        else:
            labels = [str(i) for i in range(cm.shape[0])]

        # Create figure
        figure = plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)

        # Write the image to TensorBoard
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.val_summary_writer.as_default():
            tf.summary.image('Confusion Matrix', image, step=step)

    def log_sample_predictions(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        predicted_probs: np.ndarray,
        step: int,
        max_images: int = 9
    ) -> None:
        """
        Log sample predictions to TensorBoard.

        Args:
            images: Batch of images
            true_labels: Ground truth labels (indices)
            predicted_labels: Predicted labels (indices)
            predicted_probs: Prediction probabilities
            step: Current step/epoch
            max_images: Maximum number of images to log
        """
        # Limit the number of images
        n = min(len(images), max_images)
        images = images[:n]
        true_labels = true_labels[:n]
        predicted_labels = predicted_labels[:n]
        predicted_probs = predicted_probs[:n]

        # Use class names if available
        if self.class_names is not None:
            true_class_names = [self.class_names[i] for i in true_labels]
            pred_class_names = [self.class_names[i] for i in predicted_labels]
        else:
            true_class_names = [str(i) for i in true_labels]
            pred_class_names = [str(i) for i in predicted_labels]

        # Create a grid of sample predictions
        figure = plt.figure(figsize=(12, 12))

        # Plot each image
        for i in range(n):
            plt.subplot(3, 3, i + 1)

            # Denormalize the image if necessary (if values are between 0 and 1)
            img = images[i].copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            plt.imshow(img)

            # Set color based on correctness
            color = 'green' if true_labels[i] == predicted_labels[i] else 'red'

            # Create title
            title = f"True: {true_class_names[i]}\nPred: {pred_class_names[i]}"
            plt.title(title, color=color)
            plt.axis('off')

        plt.tight_layout()

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)

        # Write the image to TensorBoard
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.val_summary_writer.as_default():
            tf.summary.image('Sample Predictions', image, step=step)

    def log_learning_rate(self, optimizer, step: int) -> None:
        """
        Log current learning rate to TensorBoard.

        Args:
            optimizer: Optimizer instance
            step: Current step/epoch
        """
        # Extract learning rate from optimizer
        if hasattr(optimizer, 'learning_rate'):
            if hasattr(optimizer.learning_rate, 'numpy'):
                lr = optimizer.learning_rate.numpy()
            elif callable(optimizer.learning_rate):
                lr = optimizer.learning_rate(step).numpy()
            else:
                lr = optimizer.learning_rate
        else:
            # Default value if learning rate cannot be extracted
            lr = 0.001

        with self.train_summary_writer.as_default():
            tf.summary.scalar('learning_rate', lr, step=step)

    def log_pr_curves(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        step: int,
        num_thresholds: int = 200
    ) -> None:
        """
        Log precision-recall curves to TensorBoard.

        Args:
            labels: One-hot encoded ground truth labels
            predictions: Predicted probabilities
            step: Current step/epoch
            num_thresholds: Number of thresholds for PR curve
        """
        # Log PR curve for each class
        with self.val_summary_writer.as_default():
            for i in range(labels.shape[1]):
                class_name = self.class_names[i] if self.class_names else f"Class_{i}"

                tf.summary.pr_curve(
                    name=f'PR_Curve/{class_name}',
                    labels=labels[:, i],
                    predictions=predictions[:, i],
                    num_thresholds=num_thresholds,
                    step=step
                )

    def log_model_weights_histogram(self, model: tf.keras.Model, step: int) -> None:
        """
        Log histograms of model weights.

        Args:
            model: Keras model
            step: Current step/epoch
        """
        with self.train_summary_writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:
                    if tf.rank(weight) > 0:  # Skip scalar weights
                        tf.summary.histogram(
                            f"{layer.name}/{weight.name}",
                            weight,
                            step=step
                        )
