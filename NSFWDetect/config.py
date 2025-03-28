"""Configuration for the NSFWDetect model and training pipeline."""
import os
from enum import Enum, auto

class NSFWCategory(Enum):
    NEUTRAL_GENERIC = auto()  # Normal SFW images
    NEUTRAL_DRAWING = auto()  # Neutral drawings
    SEXY = auto()             # Borderline NSFW (bikini photos, etc.)
    PORN = auto()             # Pornographic content
    HENTAI = auto()           # Animated/drawn pornographic content
    ARTIFICIAL = auto()       # NSFW 3D animations
    BEASTIALITY = auto()      # Animal-related NSFW content
    FURRY = auto()            # Furry pornographic content
    GORE = auto()             # Violent/gory content
    TOYS = auto()             # Sex toys in images
    VINTAGE = auto()          # Old-school pornographic content
    WTF = auto()              # Extreme/disturbing NSFW content

# Mapping from enum to string labels
CATEGORY_MAPPING = {
    NSFWCategory.NEUTRAL_GENERIC: "neutral_generic",
    NSFWCategory.NEUTRAL_DRAWING: "neutral_drawing",
    NSFWCategory.SEXY: "sexy",
    NSFWCategory.PORN: "porn",
    NSFWCategory.HENTAI: "hentai",
    NSFWCategory.ARTIFICIAL: "artificial",
    NSFWCategory.BEASTIALITY: "beastiality",
    NSFWCategory.FURRY: "furry",
    NSFWCategory.GORE: "gore",
    NSFWCategory.TOYS: "toys",
    NSFWCategory.VINTAGE: "vintage",
    NSFWCategory.WTF: "wtf",
}

# Model settings
# Using an enhanced, efficient variant of ViT available in 2025
ViT_MODEL_TYPE = "ViT-E/16"  # Efficient Vision Transformer with 16x16 patches
IMAGE_SIZE = 384  # Using higher resolution for better detection of small details
PATCH_SIZE = 16
NUM_CLASSES = len(NSFWCategory)
HIDDEN_SIZE = 1024
MLP_SIZE = HIDDEN_SIZE * 4
NUM_HEADS = 16
NUM_LAYERS = 16
DROPOUT_RATE = 0.1

# Training settings
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10000
CHECKPOINT_DIR = os.path.join("checkpoints")
TENSORBOARD_DIR = os.path.join("logs")

# Data settings
DATA_DIR = os.path.join("data", "nsfw_dataset")
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
