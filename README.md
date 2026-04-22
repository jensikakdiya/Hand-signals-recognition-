# ✋ Dual Mode Hand Gesture & Sign Language Recognition

A real-time computer vision system that recognizes **hand gestures** and **sign language symbols** using **MediaPipe** and **Machine Learning** — directly from your webcam.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn&logoColor=white)

---

## 📌 Project Overview

This project builds a real-time gesture and sign language recognition system using **MediaPipe hand landmark detection** combined with **machine learning classifiers**. It supports two independent modes — gesture recognition and sign language recognition — switchable live via keyboard.

**Use cases:**
- Assistive technology for the hearing impaired
- Touchless human-computer interaction
- Sign language learning tool

---

## 🗂️ Project Structure

```
Hand-signals-recognition/
├── collect_data.py       # Webcam-based data collection for gestures/signs
├── train_models.py       # Train ML classifiers on collected data
├── test_model.py         # Test model on a static image
├── main.py               # Real-time webcam application (dual mode)
├── data/
│   ├── gesture_data.csv  # Collected gesture landmark data
│   └── sign_data.csv     # Collected sign language landmark data
├── gesture_model.pkl     # Saved gesture classifier
├── sign_model.pkl        # Saved sign language classifier
└── README.md
```

---

## ⚙️ How It Works

1. **MediaPipe** detects 21 hand landmarks (x, y coordinates) from webcam frames
2. Landmark coordinates are used as features for an ML classifier
3. The classifier predicts the gesture/sign label in real time
4. Prediction is displayed live on the webcam feed

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install opencv-python mediapipe scikit-learn numpy joblib
```

### 2. Collect training data
```bash
python collect_data.py
```
- Choose mode: **Gesture** or **Sign Language**
- Enter a label (e.g., `thumbs_up`, `A`, `B`)
- Show your hand to the webcam — data is saved automatically
- Collect at least **200 samples per gesture** for good accuracy
- Ensure **good lighting** and keep your hand fully visible

### 3. Train the models
```bash
python train_models.py
```
Trains ML classifiers and saves:
- `gesture_model.pkl`
- `sign_model.pkl`

### 4. (Optional) Test on an image
```bash
python test_model.py
```
- Choose model type (gesture or sign)
- Provide an image path
- View the prediction result

### 5. Run the real-time app
```bash
python main.py
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `G` | Switch to Gesture Mode |
| `S` | Switch to Sign Language Mode |
| `Q` / `ESC` | Exit |

---

## 📸 Screenshots

### Real-time Webcam Output
![Webcam Output](https://private-user-images.githubusercontent.com/142288450/555392362-e7f7c747-5d87-4e81-a27b-979f366c8580.png)

### Terminal Prediction Output
![Terminal Output](https://private-user-images.githubusercontent.com/142288450/555392886-942e6e31-5b01-4acd-8724-ff9073ece855.png)

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Landmark Detection | MediaPipe |
| Image Processing | OpenCV |
| ML Classifier | Scikit-learn |
| Data Handling | NumPy, CSV |
| Model Saving | Joblib (.pkl) |

---

## 💡 Tips for Best Results

- Collect at least **200 samples** per gesture/sign
- Use **consistent lighting** — avoid shadows on your hand
- Keep your **hand fully in frame** during collection and testing
- Place `.pkl` files in the **same folder** as `main.py`

---

## 🚀 Future Enhancements

- [ ] Voice output for recognized signs (text-to-speech)
- [ ] Streamlit / web app interface
- [ ] Deep learning upgrade (CNN/LSTM for better accuracy)
- [ ] Sentence-level sign language recognition
- [ ] Support for both hands simultaneously

---

## 👩‍💻 Author

**Jensi Kakdiya**
M.Sc. Data Science | Marwadi University, Rajkot

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/jensi-kakdiya-245585282)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/jensikakdiya)

---

⭐ If you found this project useful, please give it a star!
