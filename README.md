✋ Dual Mode Hand Gesture & Sign Language Recognition

A real-time computer vision system that recognizes hand gestures and sign language symbols using MediaPipe and Machine Learning.

🗂️ Project Structure
HandRecognitionProject/
├── collect_data.py
├── train_models.py
├── test_model.py
├── p1.py
├── data/
│   ├── gesture_data.csv
│   └── sign_data.csv
├── gesture_model.pkl
└── sign_model.pkl

⚙️ Installation
pip install opencv-python mediapipe scikit-learn numpy joblib

▶️ How to Run (Step-by-Step)
✅ Step 1: Collect Data
Run:
python collect_data.py
-Choose mode (Gesture / Sign)
-Enter label name (e.g., thumbs_up, A)
-Show your hand to the webcam
-Data is saved in data/gesture_data.csv or data/sign_data.csv

✅ Step 2: Train Models
Run:
python train_models.py
-Trains ML models 
Generates:
-gesture_model.pkl
-sign_model.pkl

✅ Step 3: Test Model (Optional)
Run:
python test_model.py
-Enter model choice (gesture or sign)
-Provide image path
-View prediction result

✅ Step 4: Run Real-Time Application
Run:
python p1.py
Keyboard Controls:
-G → Gesture Mode
-S → Sign Language Mode
-Q / ESC → Exit

📸 Screenshots

1.Web-Cam Output:


<img width="802" height="640" alt="Screenshot 2026-02-26 184214" src="https://github.com/user-attachments/assets/e7f7c747-5d87-4e81-a27b-979f366c8580" />

2.Predicted Output in Terminal:


<img width="504" height="183" alt="Screenshot 2026-02-26 184331" src="https://github.com/user-attachments/assets/942e6e31-5b01-4acd-8724-ff9073ece855" />


🧠 Notes
~collect at least 200 samples per gesture/sign
~Ensure good lighting
~Keep hand fully visible in frame
~Place .pkl files in the same folder as p1.py

🚀 Future Enhancements
~Voice output
~GUI/Web app
~Deep learning models (CNN/LSTM)
~Sentence-level sign recognition

⭐ Tech Stack

Python · OpenCV · MediaPipe · Scikit-learn
