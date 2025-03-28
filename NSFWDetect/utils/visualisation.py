"""
Visualization utilities for NSFWDetect.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from typing import List, Optional, Tuple, Dict, Union
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import io
import cv2

from config import CATEGORY_MAPPING

class NSFWVisualization:
    """
    Visualization tools for NSFW detection model analysis.

    Features:
    - Attention map visualization
    - Grad-CAM for model interpretability
    - Feature embedding visualization with t-SNE
    - Training history plots
    - Example prediction visualization
    """
    def __init__(self, model, class_names=None, output_dir="visualizations"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use class names from config if not provided
        if class_names is None:
            self.class_names = [CATEGORY_MAPPING[k] for k in sorted(CATEGORY_MAPPING.keys())]
        else:
            self.class_names = class_names

    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot training history metrics.

        Args:
            history: Dictionary of training metrics
        """
        plt.figure(figsize=(18, 6))

        # Plot training & validation accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot training & validation loss
        plt.subplot(1, 3, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot training & validation F1 score
        plt.subplot(1, 3, 3)
        plt.plot(history['train_f1'], label='Train F1')
        plt.plot(history['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300)
        plt.close()

        logging.info(f"Training history plot saved to {self.output_dir / 'training_history.png'}")

    def visualize_attention_maps(self, image: np.ndarray) -> None:
        """
        Visualize attention maps from the transformer layers.

        Args:
            image: Input image as numpy array of shape [height, width, channels]
        """
        # Preprocess image
        preprocessed_image = tf.expand_dims(
            tf.image.resize(image, [self.model.image_size, self.model.image_size]),
            0
        )

        # Create model that outputs attention weights
        # For this we need to create a custom model with access to attention weights
        # This is a placeholder implementation that would need to be adapted based on the model architecture

        # Get attention maps from the last layer
        with tf.GradientTape() as tape:
            # Add a hook to extract attention maps
            # This is model-specific and would need to be implemented based on the ViT implementation
            attention_maps = self._extract_attention_maps(preprocessed_image)

        # Reshape attention maps for visualization
        # Assuming attention_maps shape: [batch_size, num_heads, num_patches, num_patches]
        attention_maps = tf.reduce_mean(attention_maps, axis=1)  # Average over heads
        attention_maps = attention_maps[0]  # Take the first (and only) batch

        # Number of patches in each dimension
        n = int(np.sqrt(attention_maps.shape[0] - 1))  # Subtract 1 for the class token

        # Plot original image
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # Plot attention map (use CLS token's attention to all patches)
        plt.subplot(1, 2, 2)
        cls_attention = attention_maps[0, 1:]  # CLS token's attention to image patches
        attention_map = cls_attention.numpy().reshape(n, n)

        # Resize attention map to image size for overlay
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Create heatmap
        attention_heatmap = cv2.applyColorMap(
            np.uint8(255 * attention_map),
            cv2.COLORMAP_JET
        )
        attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB)

        # Overlay attention map on original image
        alpha = 0.6
        overlay = alpha * attention_heatmap + (1 - alpha) * image
        overlay = overlay / overlay.max()

        plt.imshow(overlay)
        plt.title('Attention Map')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_visualization.png', dpi=300)
        plt.close()

        logging.info(f"Attention map visualization saved to {self.output_dir / 'attention_visualization.png'}")

    def _extract_attention_maps(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Extract attention maps from the model.
        This is a placeholder method that would need to be implemented based on the model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Attention maps tensor
        """
        # In a real implementation, this would extract attention weights from the model
        # For now, we'll return a dummy tensor
        batch_size = inputs.shape[0]
        num_patches = (self.model.image_size // self.model.patch_size) ** 2 + 1  # +1 for CLS token
        num_heads = self.model.num_heads

        # Create dummy attention maps
        return tf.random.uniform([batch_size, num_heads, num_patches, num_patches])

    def generate_grad_cam(self, image: np.ndarray) -> None:
        """
        Generate Grad-CAM visualization for the given image.

        Args:
            image: Input image as numpy array of shape [height, width, channels]
        """
        # Preprocess image
        preprocessed_image = tf.expand_dims(
            tf.image.resize(image, [self.model.image_size, self.model.image_size]),
            0
        )

        # Create a model that outputs both the predictions and the last convolutional layer
        # For ViT, we can use the patch embeddings before the attention layers

        # This is a placeholder implementation
        # In reality, this would need to be implemented based on the model architecture

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(preprocessed_image, training=False)

            # Get the predicted class
            pred_class = tf.argmax(predictions[0])

            # Get the output for the predicted class
            class_output = predictions[0, pred_class]

        # Get gradients of the output with respect to the last conv layer
        # In ViT, we need to identify which layer to use for visualization
        # For simplicity, let's assume we're using the output of the patch encoder

        # This is just a placeholder - in a real implementation this would be the actual layer
        last_conv_layer = self.model.patch_encoder

        # Get gradients
        grads = tape.gradient(class_output, last_conv_layer.output)

        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute the Grad-CAM by weighting the feature maps with the gradients
        # The output of the patch encoder is of shape [batch_size, num_patches, embedding_dim]
        # We need to reshape it to [batch_size, height, width, embedding_dim]
        n = int(np.sqrt(last_conv_layer.output.shape[1]))
        feature_maps = tf.reshape(last_conv_layer.output, [-1, n, n, last_conv_layer.output.shape[-1]])

        # Apply the gradients to the feature maps
        cam = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)

        # Normalize the CAM
        cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
        cam = cam[0].numpy()

        # Resize the CAM to the original image size
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on original image
        alpha = 0.6
        overlay = alpha * heatmap + (1 - alpha) * image
        overlay = overlay / overlay.max()

        # Plot
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Grad-CAM')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'grad_cam.png', dpi=300)
        plt.close()

        logging.info(f"Grad-CAM visualization saved to {self.output_dir / 'grad_cam.png'}")

    def visualize_embeddings(self, dataset: tf.data.Dataset, num_samples: int = 1000) -> None:
        """
        Visualize embeddings using t-SNE.

        Args:
            dataset: TensorFlow dataset
            num_samples: Number of samples to use for visualization
        """
        # Collect embeddings and labels
        embeddings = []
        labels = []

        # Define a model that outputs the embeddings
        # This is a placeholder that would need to be implemented based on the actual model
        embedding_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-2].output  # Assuming the second to last layer is the embedding
        )

        # Collect embeddings and labels from dataset
        sample_count = 0
        for batch in dataset:
            if sample_count >= num_samples:
                break

            images, batch_labels = batch
            batch_embeddings = embedding_model(images, training=False)

            embeddings.append(batch_embeddings.numpy())
            labels.append(np.argmax(batch_labels.numpy(), axis=1))

            sample_count += images.shape[0]

        # Concatenate all embeddings and labels
        embeddings = np.concatenate(embeddings, axis=0)[:num_samples]
        labels = np.concatenate(labels, axis=0)[:num_samples]

        # Apply t-SNE
        logging.info("Applying t-SNE to embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot the embeddings
        plt.figure(figsize=(12, 10))

        # Create a scatter plot for each class
        cmap = plt.cm.get_cmap('tab20', len(self.class_names))
        for i, class_name in enumerate(self.class_names):
            idx = labels == i
            plt.scatter(
                embeddings_2d[idx, 0],
                embeddings_2d[idx, 1],
                c=[cmap(i)],
                label=class_name,
                alpha=0.7,
                s=20
            )

        plt.title('t-SNE Visualization of Embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embeddings_tsne.png', dpi=300)
        plt.close()

        logging.info(f"Embedding visualization saved to {self.output_dir / 'embeddings_tsne.png'}")

    def visualize_predictions(self, images: List[np.ndarray], true_labels: Optional[List[int]] = None) -> None:
        """
        Visualize model predictions on a set of images.

        Args:
            images: List of input images
            true_labels: Optional list of true label indices
        """
        num_images = len(images)
        rows = int(np.ceil(num_images / 3))

        plt.figure(figsize=(18, 6 * rows))

        for i, image in enumerate(images):
            # Preprocess image
            preprocessed_image = tf.expand_dims(
                tf.image.resize(image, [self.model.image_size, self.model.image_size]),
                0
            )

            # Get predictions
            predictions = self.model(preprocessed_image, training=False)
            predictions = tf.nn.softmax(predictions)[0].numpy()

            # Get top 3 predictions
            top_k_indices = np.argsort(predictions)[::-1][:3]
            top_k_values = predictions[top_k_indices]
            top_k_labels = [self.class_names[idx] for idx in top_k_indices]

            # Plot image
            plt.subplot(rows, 3, i + 1)
            plt.imshow(image)

            # Set title based on prediction
            title = f"Prediction: {top_k_labels[0]}"
            if true_labels is not None:
                title += f"\nTrue: {self.class_names[true_labels[i]]}"
                # Highlight if prediction is correct
                if top_k_indices[0] == true_labels[i]:
                    plt.gca().spines['bottom'].set_color('green')
                    plt.gca().spines['top'].set_color('green')
                    plt.gca().spines['right'].set_color('green')
                    plt.gca().spines['left'].set_color('green')
                    plt.gca().spines['bottom'].set_linewidth(5)
                    plt.gca().spines['top'].set_linewidth(5)
                    plt.gca().spines['right'].set_linewidth(5)
                    plt.gca().spines['left'].set_linewidth(5)

            plt.title(title)
            plt.axis('off')

            # Add predictions as text
            prediction_text = ""
            for j in range(3):
                prediction_text += f"{top_k_labels[j]}: {top_k_values[j]*100:.2f}%\n"

            plt.figtext(
                0.01 + (i % 3) * 0.33,
                0.01 + (i // 3) * (1 / rows),
                prediction_text,
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions.png', dpi=300)
        plt.close()

        logging.info(f"Prediction visualization saved to {self.output_dir / 'predictions.png'}")

    def visualize_with_tensorboard(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        log_dir: str = "logs/visualizations",
        num_samples: int = 50
    ):
        """
        Create advanced visualizations of the model using TensorBoard.

        Args:
            model: Model to visualize
            dataset: Dataset for sample predictions
            log_dir: Log directory for TensorBoard
            num_samples: Number of samples to visualize
        """
        from utils.tensorboard import NSFWTensorBoard

        # Initialize TensorBoard
        tensorboard = NSFWTensorBoard(
            log_dir=log_dir,
            experiment_name=f"model_visualization",
            histogram_freq=1
        )

        # Set class names
        tensorboard.set_class_names(self.class_names)

        # Write model graph
        tensorboard._write_model_graph(model)

        # Process samples for prediction visualization
        all_images = []
        all_labels = []
        all_predictions = []
        all_pred_probs = []

        sample_count = 0
        for images, labels in dataset:
            # Get predictions
            predictions = model(images, training=False)
            pred_probs = tf.nn.softmax(predictions)
            pred_classes = tf.argmax(pred_probs, axis=1)
            true_classes = tf.argmax(labels, axis=1)

            # Store data
            all_images.append(images.numpy())
            all_labels.append(true_classes.numpy())
            all_predictions.append(pred_classes.numpy())
            all_pred_probs.append(pred_probs.numpy())

            # Update count
            sample_count += images.shape[0]

            if sample_count >= num_samples:
                break

        # Concatenate and trim
        all_images = np.concatenate(all_images, axis=0)[:num_samples]
        all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
        all_predictions = np.concatenate(all_predictions, axis=0)[:num_samples]
        all_pred_probs = np.concatenate(all_pred_probs, axis=0)[:num_samples]

        # Log samples
        tensorboard.log_sample_predictions(
            all_images, all_labels, all_predictions, all_pred_probs, step=0
        )

        # Log model weights
        tensorboard.log_model_weights_histogram(model, step=0)

        # Log PR curves
        one_hot_labels = tf.keras.utils.to_categorical(all_labels, num_classes=len(self.class_names))
        tensorboard.log_pr_curves(one_hot_labels, all_pred_probs, step=0)

        print(f"Visualizations created in TensorBoard log directory: {log_dir}")
        print("Start TensorBoard with:")
        print(f"tensorboard --logdir={log_dir}")
