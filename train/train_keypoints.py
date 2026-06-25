import cv2
import os
import numpy as np
import mediapipe as mp
import json
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "gestures_clean/train"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X = []
y = []

classes = sorted(os.listdir(DATA_DIR))

# -----------------------
# PROCESS DATA
# -----------------------
for label in classes:

    folder = os.path.join(DATA_DIR, label)

    if not os.path.isdir(folder):
        continue

    images = os.listdir(folder)

    print(f"\nProcessing class: {label} ({len(images)} images)")

    success = 0

    for img_name in tqdm(images, desc=label):

        path = os.path.join(folder, img_name)
        img = cv2.imread(path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        lm = result.multi_hand_landmarks[0]

        row = []
        for p in lm.landmark:
            row.extend([p.x, p.y, p.z])

        X.append(row)
        y.append(label)
        success += 1

    print(f"Detected hands: {success}/{len(images)}")

# -----------------------
# FINAL STATS
# -----------------------
print("\nTotal samples:", len(X))

# -----------------------
# TRAIN MODEL
# -----------------------
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1
)

model.fit(X, y)

# -----------------------
# SAVE
# -----------------------
joblib.dump(model, "gesture_keypoint_model.pkl")

with open("keypoint_classes.json", "w") as f:
    json.dump(sorted(list(set(y))), f)

print("\nSaved model + classes")