import os
import sys

# Disable TensorFlow / MediaPipe logs
os.environ["MEDIAPIPE_DISABLE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=filter INFO, 2=filter WARNING, 3=only ERROR
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOGGING"] = "none"

# Redirect stderr to silence absl C++ warnings
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

sys.stderr = NullWriter()

import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
GESTURE_MODEL_PATH = 'gesture_model.pkl'
SIGN_MODEL_PATH = 'sign_model.pkl'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------------------------------------
# Choose mode
# -------------------------------------------------------
print("Select Test Mode:")
print("1 - Gesture Model")
print("2 - Sign Language Model")
mode = input("Enter choice (1/2): ")

if mode == "1":
    model_path = GESTURE_MODEL_PATH
elif mode == "2":
    model_path = SIGN_MODEL_PATH
else:
    print("Invalid choice.")
    exit()

# Load model
try:
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")
except FileNotFoundError:
    print(f"❌ Model file not found: {model_path}")
    exit()

# -------------------------------------------------------
# Input image
# -------------------------------------------------------
image_path = input("Enter path of hand image (e.g., hand.jpg): ")
image = cv2.imread(image_path)

if image is None:
    print("❌ Could not read image. Check the path.")
    exit()

# -------------------------------------------------------
# Process the image
# -------------------------------------------------------
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        print("⚠️ No hand detected in the image.")
    else:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert to numpy array for model prediction
            prediction = model.predict([np.array(landmarks)])
            print(f"🧠 Predicted label: {prediction[0]}")

            cv2.putText(image, f"Prediction: {prediction[0]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# -------------------------------------------------------
# Display image
# -------------------------------------------------------
cv2.imshow("Model Test Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
