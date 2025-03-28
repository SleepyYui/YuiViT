"""
Evaluation module for NSFWDetect.
"""
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

from config import CATEGORY_MAPPING
from models.vit_model import ViTModel
from data.dataset import NSFWDataset

class NSFWEvaluator:
    """
    Evaluator class for NSFW image classification.

    Features:
    - Confusion matrix visualization
    - ROC curves and AUC calculation
    - Precision-recall curves
    - Per-class metrics
    - Misclassification analysis
    """
    def __init__(
        self,
        model: ViTModel,
        dataset: NSFWDataset,
        output_dir: str = "evaluation_results"
    ):
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get class names
        self.class_names = [CATEGORY_MAPPING[k] for k in sorted(CATEGORY_MAPPING.keys())]

        # Get test dataset
        self.test_ds = dataset.get_test_dataset()

        # Set up metrics
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_precision = tf.keras.metrics.Precision()
        self.test_recall = tf.keras.metrics.Recall()
        self.test_f1 = tf.keras.metrics.F1Score(average='weighted')

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.

        Returns:
            Dictionary of metrics
        """
        logging.info("Evaluating model on test dataset...")

        # Collect all predictions and ground truths
        all_predictions = []
        all_labels = []

        for images, labels in self.test_ds:
            predictions = self.model(images, training=False)
            predictions = tf.nn.softmax(predictions)

            # Update metrics
            self.test_accuracy.update_state(labels, predictions)
            self.test_precision.update_state(labels, predictions)
            self.test_recall.update_state(labels, predictions)
            self.test_f1.update_state(labels, predictions)

            # Store predictions and labels for later analysis
            all_predictions.append(predictions.numpy())
            all_labels.append(labels.numpy())

        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate metrics
        accuracy = self.test_accuracy.result().numpy()
        precision = self.test_precision.result().numpy()
        recall = self.test_recall.result().numpy()
        f1 = self.test_f1.result().numpy()

        # Log results
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test Precision: {precision:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")

        # Generate confusion matrix
        self._generate_confusion_matrix(all_predictions, all_labels)

        # Generate ROC curves
        self._generate_roc_curves(all_predictions, all_labels)

        # Generate precision-recall curves
        self._generate_pr_curves(all_predictions, all_labels)

        # Generate classification report
        self._generate_classification_report(all_predictions, all_labels)

        # Analyze misclassifications
        self._analyze_misclassifications(all_predictions, all_labels)

        # Return metrics
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def _generate_confusion_matrix(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """Generate and save confusion matrix visualization."""
        # Convert predictions and labels to class indices
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()

        logging.info(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")

    def _generate_roc_curves(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """Generate and save ROC curves for each class."""
        plt.figure(figsize=(12, 10))

        # Calculate ROC curve and AUC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )

        # Plot random guessing line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300)
        plt.close()

        logging.info(f"ROC curves saved to {self.output_dir / 'roc_curves.png'}")

    def _generate_pr_curves(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """Generate and save precision-recall curves for each class."""
        plt.figure(figsize=(12, 10))

        # Calculate precision-recall curve for each class
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])
            pr_auc = auc(recall, precision)

            plt.plot(
                recall,
                precision,
                lw=2,
                label=f'{class_name} (AUC = {pr_auc:.2f})'
            )

        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=300)
        plt.close()

        logging.info(f"Precision-recall curves saved to {self.output_dir / 'pr_curves.png'}")

    def _generate_classification_report(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """Generate and save classification report."""
        # Convert predictions and labels to class indices
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)

        # Generate classification report
        report = classification_report(
            true_classes,
            pred_classes,
            target_names=self.class_names,
            output_dict=True
        )

        # Convert report to DataFrame for visualization
        import pandas as pd
        df_report = pd.DataFrame(report).transpose()

        # Save report as CSV
        df_report.to_csv(self.output_dir / 'classification_report.csv')

        # Save report as styled HTML
        with open(self.output_dir / 'classification_report.html', 'w') as f:
            f.write('<html><body>\n')
            f.write('<h1>Classification Report</h1>\n')
            f.write(df_report.style.background_gradient(cmap='Blues').to_html())
            f.write('</body></html>')

        logging.info(f"Classification report saved to {self.output_dir / 'classification_report.csv'}")

    def _analyze_misclassifications(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """Analyze and log patterns in misclassifications."""
        # Convert predictions and labels to class indices
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)

        # Find misclassifications
        misclassified = (pred_classes != true_classes)
        misclassified_indices = np.where(misclassified)[0]

        # Count misclassifications per class
        misclassifications_per_class = {}
        for i in misclassified_indices:
            true_class = true_classes[i]
            pred_class = pred_classes[i]

            if true_class not in misclassifications_per_class:
                misclassifications_per_class[true_class] = {}

            if pred_class not in misclassifications_per_class[true_class]:
                misclassifications_per_class[true_class][pred_class] = 0

            misclassifications_per_class[true_class][pred_class] += 1

        # Log misclassification patterns
        with open(self.output_dir / 'misclassification_analysis.txt', 'w') as f:
            f.write("=== Misclassification Analysis ===\n\n")

            for true_class, predictions_dict in misclassifications_per_class.items():
                true_class_name = self.class_names[true_class]
                total_misclassifications = sum(predictions_dict.values())

                f.write(f"Class: {true_class_name}\n")
                f.write(f"Total misclassifications: {total_misclassifications}\n")
                f.write("Misclassified as:\n")

                for pred_class, count in sorted(
                    predictions_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    pred_class_name = self.class_names[pred_class]
                    percentage = (count / total_misclassifications) * 100
                    f.write(f"  {pred_class_name}: {count} ({percentage:.2f}%)\n")

                f.write("\n")

            # Overall statistics
            total_samples = len(true_classes)
            total_misclassified = len(misclassified_indices)
            accuracy = (1 - (total_misclassified / total_samples)) * 100

            f.write(f"Overall Statistics:\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Total correctly classified: {total_samples - total_misclassified}\n")
            f.write(f"Total misclassified: {total_misclassified}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")

        logging.info(f"Misclassification analysis saved to {self.output_dir / 'misclassification_analysis.txt'}")
