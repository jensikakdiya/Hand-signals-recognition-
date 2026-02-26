import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_DIR = "data"
GESTURE_FILE = os.path.join(DATA_DIR, "gesture_data.csv")
SIGN_FILE = os.path.join(DATA_DIR, "sign_data.csv")

# -------------------------------------------------------
# Train model function
# -------------------------------------------------------
def train_and_save_model(csv_file, model_name):
    if not os.path.exists(csv_file):
        print(f"⚠️ File not found: {csv_file}")
        return
    
    print(f"📦 Loading data from {csv_file} ...")
    data = pd.read_csv(csv_file, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print("🧠 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_name)
    print(f"✅ Model saved as {model_name}")

# -------------------------------------------------------
# Train both models
# -------------------------------------------------------
train_and_save_model(GESTURE_FILE, 'gesture_model.pkl')
train_and_save_model(SIGN_FILE, 'sign_model.pkl')
