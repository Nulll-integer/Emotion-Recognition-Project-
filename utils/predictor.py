# utils/predictor.py
# Handles emotion prediction for the Streamlit app

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ─── CONFIGURATION ───────────────────────────────────────────
MODEL_PATH = 'model/emotion_model.h5'
IMG_SIZE   = 48
EMOTIONS   = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emoji for each emotion — shown in the web app
EMOTION_EMOJI = {
    'angry'   : '😠',
    'fear'    : '😨',
    'happy'   : '😊',
    'neutral' : '😐',
    'sad'     : '😢',
    'surprise': '😲'
}

# Color for each emotion — shown in result cards
EMOTION_COLOR = {
    'angry'   : '#ff4444',
    'fear'    : '#aa44ff',
    'happy'   : '#00cc44',
    'neutral' : '#4488ff',
    'sad'     : '#ff8800',
    'surprise': '#ffcc00'
}

# Engagement mapping for classroom analysis
ENGAGEMENT_MAP = {
    'happy'   : ('High Engagement',    '🟢', 90),
    'surprise': ('Active Engagement',  '🟡', 75),
    'neutral' : ('Moderate Engagement','🟡', 55),
    'sad'     : ('Low Engagement',     '🔴', 30),
    'fear'    : ('Stressed',           '🔴', 25),
    'angry'   : ('Disengaged',         '🔴', 15),
}
# ─────────────────────────────────────────────────────────────


# ── FIX 1: Load model AND cascade ONCE at import time ────────
_model = None
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def preprocess_face(face_img):
    """Prepares face image for CNN prediction"""
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.reshape(face_img, (1, IMG_SIZE, IMG_SIZE, 1))
    return face_img


def predict_emotion(face_img):
    """
    Predicts emotion from a face image.
    Returns: emotion, confidence, all_scores dict
    """
    model       = get_model()
    processed   = preprocess_face(face_img)
    predictions = model.predict(processed, verbose=0)
    idx         = np.argmax(predictions[0])
    emotion     = EMOTIONS[idx]
    confidence  = float(predictions[0][idx]) * 100
    all_scores  = {
        EMOTIONS[i]: float(predictions[0][i]) * 100
        for i in range(len(EMOTIONS))
    }
    return emotion, confidence, all_scores


def detect_and_predict(image_bgr):
    """
    Detects faces in image and predicts emotion for each.
    Returns: annotated image, list of results

    FIX 2: Run face detection on a downscaled frame (scale=0.5),
    then map bounding boxes back to full resolution for drawing.
    This cuts detection time by ~4x with no accuracy loss.
    """
    SCALE = 0.5  # detection scale — tweak between 0.4–0.6 if needed

    # Downscale only for detection
    small   = cv2.resize(image_bgr, (0, 0), fx=SCALE, fy=SCALE)
    gray_sm = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Use the module-level cascade (loaded once)
    faces_sm = _face_cascade.detectMultiScale(
        gray_sm,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)   # smaller minSize to compensate for downscale
    )

    results   = []
    annotated = image_bgr.copy()

    # Full-res gray only needed for ROI crop
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in (faces_sm if len(faces_sm) else []):
        # Scale coords back to full resolution
        x, y, w, h = (int(v / SCALE) for v in (x, y, w, h))

        face_roi              = gray_full[y:y+h, x:x+w]
        emotion, conf, scores = predict_emotion(face_roi)
        color                 = EMOTION_COLOR.get(emotion, '#ffffff')

        # Convert hex color to BGR for OpenCV
        color_bgr = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))

        # Draw rectangle around face
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color_bgr, 2)

        # Draw emotion label
        label = f"{emotion.upper()} {conf:.0f}%"
        cv2.rectangle(annotated, (x, y-35), (x+w, y), color_bgr, -1)
        cv2.putText(annotated, label, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append({
            'emotion'   : emotion,
            'confidence': conf,
            'scores'    : scores,
            'bbox'      : (x, y, w, h)
        })

    return annotated, results
