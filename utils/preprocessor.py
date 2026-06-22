# utils/preprocessor.py
# Handles all data preprocessing for the emotion recognition system

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# ─── CONFIGURATION ───────────────────────────────────────────
IMG_SIZE    = 48          # FER-2013 images are 48x48 pixels
BATCH_SIZE  = 64          # Number of images processed at once
NUM_CLASSES = 6           # angry, fear, happy, neutral, sad, surprise

EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

TRAIN_PATH = 'dataset/train'
TEST_PATH  = 'dataset/test'
# ─────────────────────────────────────────────────────────────


def get_data_generators():
    """
    Creates and returns train and test data generators.
    - Training data: augmented + normalized
    - Test data: normalized only (no augmentation)
    """

    # ── Training Generator (with augmentation) ──────────────
    # WHY augmentation? To artificially increase dataset variety
    # so the model generalizes better to real-world faces
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalize: 0-255 → 0.0-1.0
        rotation_range=15,         # Randomly rotate images ±15°
        width_shift_range=0.1,     # Randomly shift horizontally
        height_shift_range=0.1,    # Randomly shift vertically
        horizontal_flip=True,      # Mirror images left-right
        zoom_range=0.1,            # Random zoom in/out
        fill_mode='nearest'        # Fill empty pixels after transforms
    )

    # ── Test Generator (NO augmentation — only normalize) ───
    # WHY no augmentation for test? We evaluate on real,
    # unmodified images to get honest performance metrics
    test_datagen = ImageDataGenerator(
        rescale=1./255             # Only normalize, nothing else
    )

    # ── Load Training Data ───────────────────────────────────
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize all to 48x48
        color_mode='grayscale',            # FER-2013 is grayscale
        batch_size=BATCH_SIZE,
        class_mode='categorical',          # One-hot encoded labels
        classes=EMOTIONS,                  # Our 6 emotion classes
        shuffle=True                       # Shuffle for better training
    )

    # ── Load Test Data ───────────────────────────────────────
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=EMOTIONS,
        shuffle=False                      # Don't shuffle test data
    )

    return train_generator, test_generator


def get_class_labels():
    """Returns the emotion class labels"""
    return EMOTIONS