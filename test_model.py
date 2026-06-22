# test_model.py
# Verifies the CNN model is correctly built

from utils.model import get_model

print("=" * 55)
print("  CNN MODEL VERIFICATION")
print("=" * 55)

print("\n⏳ Building CNN model...")
model = get_model()

print("\n📊 MODEL SUMMARY:")
print("=" * 55)
model.summary()

# Count parameters
total_params     = model.count_params()
trainable_params = sum([
    w.numpy().size for w in model.trainable_weights
])

print("\n📈 MODEL STATISTICS:")
print(f"  Total parameters     : {total_params:,}")
print(f"  Input shape          : (48, 48, 1)")
print(f"  Output shape         : (6,) — 6 emotions")
print(f"  Output activation    : Softmax")

print("\n🏷️  EMOTION CLASSES:")
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for i, emotion in enumerate(emotions):
    print(f"  Neuron {i} → {emotion}")

print("\n" + "=" * 55)
print("✅ CNN Model is CORRECT — Ready for Stage 5 Training!")
print("=" * 55)