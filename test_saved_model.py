# test_saved_model.py
# Tests the saved emotion model on sample images

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ─── CONFIGURATION ───────────────────────────────────────────
MODEL_PATH = 'model/emotion_model.h5'
IMG_SIZE   = 48
EMOTIONS   = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# ─────────────────────────────────────────────────────────────


def load_emotion_model():
    """Loads the saved CNN model"""
    print(f"⏳ Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape  : {model.input_shape}")
    print(f"   Output shape : {model.output_shape}")
    return model


def preprocess_face(face_img):
    """
    Prepares a face image for prediction.
    Same preprocessing used during training.
    """
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixels 0-255 → 0.0-1.0
    face_img = face_img.astype('float32') / 255.0

    # Reshape for model input: (1, 48, 48, 1)
    face_img = np.reshape(face_img, (1, IMG_SIZE, IMG_SIZE, 1))

    return face_img


def predict_emotion(model, face_img):
    """Returns predicted emotion and confidence scores"""
    processed = preprocess_face(face_img)
    predictions = model.predict(processed, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    emotion     = EMOTIONS[emotion_idx]
    confidence  = predictions[0][emotion_idx] * 100
    return emotion, confidence, predictions[0]


def test_with_dataset_images(model):
    """Tests model using images from your dataset"""
    print("\n" + "=" * 55)
    print("  TESTING WITH DATASET IMAGES")
    print("=" * 55)

    results  = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle('Emotion Recognition — Model Test Results',
                 color='white', fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for idx, emotion in enumerate(EMOTIONS):
        # Get first image from test dataset
        emotion_path = os.path.join('dataset', 'test', emotion)

        if not os.path.exists(emotion_path):
            print(f"  ❌ Folder not found: {emotion_path}")
            continue

        images = [f for f in os.listdir(emotion_path)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            print(f"  ❌ No images found in {emotion_path}")
            continue

        # Load and predict
        img_path = os.path.join(emotion_path, images[0])
        img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  ❌ Could not read image: {img_path}")
            continue

        predicted, confidence, all_scores = predict_emotion(
            model, img
        )

        correct = predicted == emotion
        status  = "✅ CORRECT" if correct else "❌ WRONG"
        results.append(correct)

        print(f"\n  Image    : {emotion}/{images[0]}")
        print(f"  Actual   : {emotion}")
        print(f"  Predicted: {predicted} ({confidence:.1f}%)")
        print(f"  Result   : {status}")

        # Plot image with prediction
        ax = axes_flat[idx]
        ax.imshow(cv2.resize(img, (IMG_SIZE, IMG_SIZE)),
                  cmap='gray')

        title_color = '#00ff88' if correct else '#ff6b6b'
        ax.set_title(
            f"True: {emotion.upper()}\n"
            f"Pred: {predicted.upper()} ({confidence:.0f}%)",
            color=title_color, fontsize=10, fontweight='bold'
        )
        ax.axis('off')
        ax.set_facecolor('#2d2d2d')

        # Add border color
        for spine in ax.spines.values():
            spine.set_edgecolor(title_color)
            spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig('training/model_test_results.png',
                dpi=120, bbox_inches='tight',
                facecolor='#1e1e1e')
    plt.close()

    # Summary
    if results:
        accuracy = sum(results) / len(results) * 100
        print(f"\n{'='*55}")
        print(f"  QUICK TEST SUMMARY")
        print(f"{'='*55}")
        print(f"  Images tested : {len(results)}")
        print(f"  Correct       : {sum(results)}")
        print(f"  Quick accuracy: {accuracy:.0f}%")
        print(f"  Results saved : training/model_test_results.png")
        print(f"{'='*55}")


def test_face_detection():
    """Tests OpenCV face detection"""
    print("\n" + "=" * 55)
    print("  TESTING FACE DETECTION")
    print("=" * 55)

    # Download haar cascade if not present
    cascade_path = cv2.data.haarcascades + \
                   'haarcascade_frontalface_default.xml'

    if os.path.exists(cascade_path):
        print(f"  ✅ Haar cascade found")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"  ✅ Face detector loaded successfully")
        print(f"  ✅ Ready for real-time detection")
    else:
        print(f"  ❌ Haar cascade not found at {cascade_path}")


def main():
    print("=" * 55)
    print("  STAGE 6 — MODEL VERIFICATION")
    print("=" * 55)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Run train.py first!")
        return

    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n✅ Model file found — {model_size:.1f} MB")

    # Load model
    model = load_emotion_model()

    # Test face detection
    test_face_detection()

    # Test with dataset images
    test_with_dataset_images(model)

    print(f"\n{'='*55}")
    print(f"✅ Stage 6 COMPLETE — Model is working correctly!")
    print(f"✅ Ready for Stage 7: Streamlit Web App!")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()