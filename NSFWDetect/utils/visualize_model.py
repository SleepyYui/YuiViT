#!/usr/bin/env python3
"""
Utility script to visualize the NSFWDetect model architecture and features with TensorBoard.
"""
import argparse
import tensorflow as tf
import numpy as np
import logging
import datetime
from pathlib import Path

from models.vit_model import ViTModel
from data.dataset import NSFWDataset
from utils.tensorboard import NSFWTensorBoard
from config import IMAGE_SIZE, CATEGORY_MAPPING

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize NSFWDetect model with TensorBoard")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/nsfw_dataset",
                        help="Path to dataset for sample inference")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataset")
    parser.add_argument("--log_dir", type=str, default="logs/visualization",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to process")

    return parser.parse_args()

def main():
    args = parse_args()

    # Create TensorBoard directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize TensorBoard
    tensorboard = NSFWTensorBoard(
        log_dir=args.log_dir,
        experiment_name=f"model_viz_{timestamp}",
        histogram_freq=1
    )

    # Set class names
    class_names = [CATEGORY_MAPPING[k] for k in sorted(CATEGORY_MAPPING.keys())]
    tensorboard.set_class_names(class_names)

    # Create model
    logging.info("Creating model...")
    model = ViTModel()

    # Load checkpoint if provided
    if args.checkpoint:
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(args.checkpoint).expect_partial()

    # Write model graph to TensorBoard
    logging.info("Writing model graph to TensorBoard...")
    tensorboard._write_model_graph(model)

    # Create dataset
    logging.info(f"Loading dataset from {args.data_dir}...")
    dataset = NSFWDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE
    )

    # Get a test batch for visualization
    logging.info("Processing sample images...")
    test_ds = dataset.get_test_dataset()

    # Track images and predictions
    all_images = []
    all_labels = []
    all_predictions = []
    all_pred_probs = []

    sample_count = 0
    # Process batches until we have enough samples
    for images, labels in test_ds:
        # Get predictions
        predictions = model(images, training=False)
        pred_probs = tf.nn.softmax(predictions)
        pred_classes = tf.argmax(pred_probs, axis=1)
        true_classes = tf.argmax(labels, axis=1)

        # Add batch to collections
        all_images.append(images.numpy())
        all_labels.append(true_classes.numpy())
        all_predictions.append(pred_classes.numpy())
        all_pred_probs.append(pred_probs.numpy())

        # Update sample count
        sample_count += images.shape[0]

        # Check if we have enough samples
        if sample_count >= args.num_samples:
            break

    # Concatenate arrays
    all_images = np.concatenate(all_images, axis=0)[:args.num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:args.num_samples]
    all_predictions = np.concatenate(all_predictions, axis=0)[:args.num_samples]
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)[:args.num_samples]

    # Log sample predictions
    logging.info("Logging sample predictions...")
    tensorboard.log_sample_predictions(
        all_images, all_labels, all_predictions, all_pred_probs, step=0
    )

    # Log model weights histograms
    logging.info("Logging model weights...")
    tensorboard.log_model_weights_histogram(model, step=0)

    # Start TensorBoard
    import subprocess
    import webbrowser
    import time

    logging.info(f"Starting TensorBoard at port {args.port}...")
    tensorboard_process = subprocess.Popen(
        ["tensorboard", f"--logdir={args.log_dir}", f"--port={args.port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for TensorBoard to start
    time.sleep(3)

    # Open browser
    url = f"http://localhost:{args.port}/"
    webbrowser.open(url)

    logging.info(f"TensorBoard is running at {url}")
    logging.info("Press Ctrl+C to exit...")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        tensorboard_process.terminate()

if __name__ == "__main__":
    main()
