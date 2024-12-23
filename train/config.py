from pathlib import Path

# Model Configuration
NUM_CLASSES = 120
BATCH_SIZE = 32
EPOCHS = 50

# Optimization
LEARNING_RATE = 0.008
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3
LABEL_SMOOTHING = 0.1

# Paths
MODEL_PATH = Path("saved_models")
MODEL_NAME = "doggie_resnet.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Dataset
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# TEST_SPLIT is implicitly 1 - (TRAIN_SPLIT + VAL_SPLIT)

# Data Augmentation
CROP_SIZE = 224
ROTATION_DEGREES = 15
COLOR_JITTER = {
    'brightness': 0.3,
    'contrast': 0.3,
    'saturation': 0.3,
    'hue': 0.15
}
