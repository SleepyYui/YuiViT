"""
Main script for NSFWDetect.
"""
import os
import argparse
import logging
import tensorflow as tf
from pathlib import Path

from config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_STEPS, CHECKPOINT_DIR, TENSORBOARD_DIR
)
from models.vit_model import ViTModel
from data.dataset import NSFWDataset
from training.train import NSFWTrainer
from training.evaluate import NSFWEvaluator
import tensorflow_addons as tfa

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nsfw_detect.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NSFWDetect: Train and evaluate a NSFW content detection model")
    parser.add_argument("--data_dir", type=str, default="data/nsfw_dataset", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay rate")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, default=TENSORBOARD_DIR, help="Directory for TensorBoard logs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--cache_dataset", action="store_true", help="Cache dataset in memory")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "both"], default="both",
                        help="Mode to run: train, evaluate, or both")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set up mixed precision if requested
    if args.use_mixed_precision:
        logging.info("Using mixed precision training")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    # Log TensorFlow version and GPU info
    logging.info(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"Using GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logging.warning("No GPU found, using CPU")

    # Verify TensorFlow Addons is available
    try:
        logging.info(f"TensorFlow Addons version: {tfa.__version__}")
    except:
        logging.error("TensorFlow Addons not found. Please install with: pip install tensorflow-addons")
        return

    # Create dataset
    logging.info(f"Creating dataset from {args.data_dir}")
    dataset = NSFWDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
        use_cache=args.cache_dataset
    )

    # Create model
    logging.info("Creating Vision Transformer model")
    model = ViTModel()

    # Train or evaluate based on mode
    if args.mode in ["train", "both"]:
        # Create trainer
        trainer = NSFWTrainer(
            model=model,
            dataset=dataset,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=WARMUP_STEPS,
            checkpoint_dir=args.checkpoint_dir,
            tensorboard_dir=args.tensorboard_dir,
            use_mixed_precision=args.use_mixed_precision
        )

        # Train model
        logging.info(f"Training model for {args.epochs} epochs")
        history = trainer.train(epochs=args.epochs)

        # Log final metrics
        logging.info(f"Final training metrics:")
        logging.info(f"  Loss: {history['train_loss'][-1]:.4f}")
        logging.info(f"  Accuracy: {history['train_accuracy'][-1]:.4f}")
        logging.info(f"  F1 Score: {history['train_f1'][-1]:.4f}")
        logging.info(f"Final validation metrics:")
        logging.info(f"  Loss: {history['val_loss'][-1]:.4f}")
        logging.info(f"  Accuracy: {history['val_accuracy'][-1]:.4f}")
        logging.info(f"  F1 Score: {history['val_f1'][-1]:.4f}")

    if args.mode in ["evaluate", "both"]:
        # Load checkpoint if evaluating only
        if args.mode == "evaluate" and args.checkpoint:
            logging.info(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(args.checkpoint).expect_partial()

        # Create evaluator
        evaluator = NSFWEvaluator(
            model=model,
            dataset=dataset,
            output_dir=args.output_dir
        )

        # Evaluate model
        logging.info("Evaluating model on test dataset")
        metrics = evaluator.evaluate()

        # Log evaluation metrics
        logging.info(f"Evaluation metrics:")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1 Score: {metrics['f1']:.4f}")

        # Check if model meets target accuracy
        if metrics['accuracy'] >= 0.95:
            logging.info("🎉 SUCCESS: Model meets the target accuracy of 95% or higher!")
        else:
            logging.warning(f"⚠️ Model accuracy of {metrics['accuracy']:.2%} is below the target of 95%.")
            logging.info("Consider the following actions:")
            logging.info("- Increase training epochs")
            logging.info("- Adjust learning rate or optimizer settings")
            logging.info("- Try different data augmentation strategies")
            logging.info("- Increase model capacity or try a different architecture")

if __name__ == "__main__":
    main()
