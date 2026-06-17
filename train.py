# train.py
# Main training script for Emotion Recognition CNN
# Run this file to train your model: python train.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Silence TF messages

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)

from utils.preprocessor import get_data_generators, get_class_labels
from utils.model import get_model

# ─── CONFIGURATION ───────────────────────────────────────────
EPOCHS      = 60        # Maximum training rounds
MODEL_PATH  = 'model/emotion_model.h5'
LOG_PATH    = 'training/training_log.csv'
PLOT_PATH   = 'training/training_history.png'
# ─────────────────────────────────────────────────────────────

def create_callbacks():
    """
    Callbacks automatically manage training:
    - Save best model
    - Stop early if no improvement
    - Reduce learning rate when stuck
    - Log all metrics to CSV
    """

    # Saves ONLY the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Stops training if val_accuracy doesn't improve for 15 epochs
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Reduces learning rate when training gets stuck
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # Saves all metrics to CSV for your report
    csv_logger = CSVLogger(LOG_PATH, append=False)

    return [checkpoint, early_stop, reduce_lr, csv_logger]


def plot_training_history(history):
    """
    Creates and saves training charts for your report.
    Generates accuracy and loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle('CNN Training History — Emotion Recognition',
                 color='white', fontsize=14, fontweight='bold')

    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    # ── Accuracy Plot ────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#2d2d2d')
    ax1.plot(epochs_ran, history.history['accuracy'],
             color='#00d4ff', linewidth=2, label='Train Accuracy')
    ax1.plot(epochs_ran, history.history['val_accuracy'],
             color='#ff6b6b', linewidth=2, label='Val Accuracy')
    ax1.set_title('Model Accuracy', color='white', fontweight='bold')
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.set_ylabel('Accuracy', color='#aaa')
    ax1.legend(facecolor='#3d3d3d', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555')

    # ── Loss Plot ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#2d2d2d')
    ax2.plot(epochs_ran, history.history['loss'],
             color='#00d4ff', linewidth=2, label='Train Loss')
    ax2.plot(epochs_ran, history.history['val_loss'],
             color='#ff6b6b', linewidth=2, label='Val Loss')
    ax2.set_title('Model Loss', color='white', fontweight='bold')
    ax2.set_xlabel('Epoch', color='#aaa')
    ax2.set_ylabel('Loss', color='#aaa')
    ax2.legend(facecolor='#3d3d3d', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555')

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120,
                bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"\n Training charts saved → {PLOT_PATH}")


def evaluate_model(model, test_generator):
    """Evaluates model on test data and prints results"""
    print("\n Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(
        test_generator, verbose=1
    )
    print(f"\n{'='*50}")
    print(f"  FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*50}")
    return test_accuracy, test_loss


def main():
    print("=" * 55)
    print("  EMOTION RECOGNITION — CNN TRAINING")
    print("=" * 55)

    # ── Step 1: Load Data ────────────────────────────────────
    print("\n Loading dataset...")
    train_gen, test_gen = get_data_generators()
    print(f"  Training samples : {train_gen.samples:,}")
    print(f"   Test samples     : {test_gen.samples:,}")

    # ── Step 2: Build Model ──────────────────────────────────
    print("\n Building CNN model...")
    model = get_model()
    print(f"   Model built — {model.count_params():,} parameters")

    # ── Step 3: Setup Callbacks ──────────────────────────────
    print("\n  Setting up training callbacks...")
    callbacks = create_callbacks()
    print("   ModelCheckpoint — saves best model automatically")
    print("   EarlyStopping   — stops if no improvement")
    print("   ReduceLROnPlateau — adjusts learning rate")
    print("   CSVLogger       — logs all metrics")

    # ── Step 4: Train ────────────────────────────────────────
    print(f"\n Starting training — up to {EPOCHS} epochs...")
    print("  This will take 20-40 minutes. Do not close VS Code!")
    print("-" * 55)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 5: Plot Results ─────────────────────────────────
    print("\n Generating training charts...")
    plot_training_history(history)

    # ── Step 6: Evaluate ─────────────────────────────────────
    test_acc, test_loss = evaluate_model(model, test_gen)

    # ── Step 7: Summary ──────────────────────────────────────
    best_val_acc = max(history.history['val_accuracy'])
    epochs_ran   = len(history.history['accuracy'])

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*55}")
    print(f"  Epochs completed  : {epochs_ran}")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Final test acc    : {test_acc*100:.2f}%")
    print(f"  Model saved to    : {MODEL_PATH}")
    print(f"  Training log      : {LOG_PATH}")
    print(f"  Training charts   : {PLOT_PATH}")
    print(f"{'='*55}")
    


if __name__ == '__main__':
    main()
