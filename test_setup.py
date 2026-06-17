print("Testing environment setup...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"❌ TensorFlow FAILED: {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV FAILED: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy FAILED: {e}")

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except Exception as e:
    print(f"❌ Streamlit FAILED: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"❌ Scikit-learn FAILED: {e}")

try:
    from PIL import Image
    print(f"✅ Pillow: OK")
except Exception as e:
    print(f"❌ Pillow FAILED: {e}")

