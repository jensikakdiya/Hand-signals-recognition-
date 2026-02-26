import cv2
import mediapipe as mp
import csv
import os
import time

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
DATA_DIR = "data"
GESTURE_FILE = os.path.join(DATA_DIR, "gesture_data.csv")
SIGN_FILE = os.path.join(DATA_DIR, "sign_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------------------------------------
# Helper function to collect data
# -------------------------------------------------------
def collect_data(label_name, file_name):
    cap = cv2.VideoCapture(0)
    collected = 0
    total = 200  # how many samples to collect per class

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)

            print(f"➡️ Collecting data for '{label_name}' ({total} samples)")
            time.sleep(2)

            while collected < total:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        # Write row: all landmarks + label
                        writer.writerow(landmarks + [label_name])
                        collected += 1

                        cv2.putText(frame, f"Collecting: {label_name} ({collected}/{total})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"✅ Done collecting {collected} samples for '{label_name}'")

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------
# Main Section
# -------------------------------------------------------
print("Select Mode:")
print("1 - Gesture Data")
print("2 - Sign Language Data")
mode_choice = input("Enter choice (1/2): ")

if mode_choice == "1":
    file_name = GESTURE_FILE
elif mode_choice == "2":
    file_name = SIGN_FILE
else:
    print("Invalid choice.")
    exit()

label = input("Enter label name (e.g., thumbs_up / A / Hello): ")
collect_data(label, file_name)
