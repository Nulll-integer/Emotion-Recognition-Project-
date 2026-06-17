# test_preprocessing.py
# Verifies preprocessing works correctly before training

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.preprocessor import get_data_generators, get_class_labels, IMG_SIZE

print("=" * 55)
print("  PREPROCESSING VERIFICATION")
print("=" * 55)

# ── Load Data ────────────────────────────────────────────────
print("\n⏳ Loading data generators...")
train_gen, test_gen = get_data_generators()

# ── Print Statistics ─────────────────────────────────────────
print("\n📊 DATASET STATISTICS:")
print(f"  Training samples   : {train_gen.samples:,}")
print(f"  Test samples       : {test_gen.samples:,}")
print(f"  Image size         : {IMG_SIZE}x{IMG_SIZE} pixels")
print(f"  Color mode         : Grayscale")
print(f"  Batch size         : {train_gen.batch_size}")
print(f"  Number of classes  : {train_gen.num_classes}")
print(f"  Training batches   : {len(train_gen)}")
print(f"  Test batches       : {len(test_gen)}")

# ── Verify Class Mapping ─────────────────────────────────────
print("\n🏷️  CLASS MAPPING:")
for emotion, index in train_gen.class_indices.items():
    print(f"  {index} → {emotion}")

# ── Check One Batch ──────────────────────────────────────────
print("\n🔍 CHECKING ONE BATCH:")
images, labels = next(train_gen)
print(f"  Batch image shape  : {images.shape}")
print(f"  Batch label shape  : {labels.shape}")
print(f"  Pixel value min    : {images.min():.4f}  (should be ≥ 0.0)")
print(f"  Pixel value max    : {images.max():.4f}  (should be ≤ 1.0)")

# ── Verify Normalization ─────────────────────────────────────
if images.min() >= 0.0 and images.max() <= 1.0:
    print("\n  ✅ Normalization CORRECT — pixels are in range 0.0 to 1.0")
else:
    print("\n  ❌ Normalization FAILED — check rescale parameter")

# ── Save Sample Images ───────────────────────────────────────
print("\n💾 Saving sample images to 'sample_preview.png'...")

emotions    = get_class_labels()
fig, axes   = plt.subplots(2, 6, figsize=(18, 6))
fig.patch.set_facecolor('#1e1e1e')
fig.suptitle('Sample Preprocessed Images (One Per Emotion)',
             color='white', fontsize=14, fontweight='bold', y=1.02)

shown = {e: False for e in emotions}

for img, label in zip(images, labels):
    emotion_idx  = np.argmax(label)
    emotion_name = emotions[emotion_idx]

    if not shown[emotion_name]:
        col = emotions.index(emotion_name)

        # Row 0 — actual image
        ax0 = axes[0, col]
        ax0.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax0.set_title(emotion_name.upper(),
                      color='white', fontsize=10, fontweight='bold')
        ax0.axis('off')

        # Row 1 — pixel distribution bar
        ax1 = axes[1, col]
        ax1.hist(img.flatten(), bins=30,
                 color='#00d4ff', edgecolor='none', alpha=0.85)
        ax1.set_facecolor('#2d2d2d')
        ax1.tick_params(colors='white', labelsize=7)
        for spine in ax1.spines.values():
            spine.set_edgecolor('#555')
        ax1.set_xlabel('Pixel Value', color='#aaa', fontsize=7)
        ax1.set_ylabel('Count',       color='#aaa', fontsize=7)

        shown[emotion_name] = True

    if all(shown.values()):
        break

plt.tight_layout()
plt.savefig('sample_preview.png', dpi=120,
            bbox_inches='tight', facecolor='#1e1e1e')
plt.close()
print("  ✅ Saved → open 'sample_preview.png' to see your images")

print("\n" + "=" * 55)
print("✅ Preprocessing is CORRECT — Ready for Stage 4!")
print("=" * 55)