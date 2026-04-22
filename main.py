import cv2
import mediapipe as mp
import numpy as np
import joblib

# ------------------------------------------------------
# 1. Load Models (Optional: comment out if not trained yet)
# ------------------------------------------------------
try:
    gesture_model = joblib.load('gesture_model.pkl')
    sign_model = joblib.load('sign_model.pkl')
    print("✅ Models loaded successfully.")
except:
    gesture_model = None
    sign_model = None
    print("⚠️ Models not found. Running in demo mode (no predictions).")

# ------------------------------------------------------
# 2. Initialize MediaPipe and Webcam
# ------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

mode = 'gesture'  # Default mode

last_prediction = ""

# ------------------------------------------------------
# 3. Main Loop
# ------------------------------------------------------
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction_text = ""

        # ------------------------------------------------------
        # 4. Hand Landmark Detection
        # ------------------------------------------------------
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark features
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks).reshape(1, -1)

                # ------------------------------------------------------
                # 5. Prediction Based on Mode
                # ------------------------------------------------------
                if mode == 'gesture' and gesture_model is not None:
                    prediction = gesture_model.predict(landmarks)
                    prediction_text = f"Gesture: {prediction[0]}"
                elif mode == 'sign' and sign_model is not None:
                    prediction = sign_model.predict(landmarks)
                    prediction_text = f"Sign: {prediction[0]}"
                else:
                    prediction_text = "Model not loaded."

        # ------------------------------------------------------
        # 6. Display Current Mode and Prediction
        # ------------------------------------------------------
        cv2.putText(frame, f"MODE: {mode.upper()}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, prediction_text,
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press G : Gesture Mode",
            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Press S : Sign Mode",
            (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Press Esc : Quit",
            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Print to terminal only when prediction changes
        if prediction_text != last_prediction:
            print("Predicted:", prediction_text)
            last_prediction = prediction_text

        cv2.imshow("Hand Gesture & Sign Recognition", frame)
        # ------------------------------------------------------
        # 7. Keyboard Input for Mode Switching
        # ------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            mode = 'gesture'
        elif key == ord('s'):
            mode = 'sign'
        elif key == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()
