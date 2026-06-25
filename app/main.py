import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import time
from flask import Flask, Response, render_template_string
from collections import deque

# =========================
# APP
# =========================
app = Flask(__name__)

# =========================
# MODEL
# =========================
import os

BASE_DIR = "/home/nvidia1/nvidia-gesture-control"

model = joblib.load(os.path.join(BASE_DIR, "models/gesture_keypoint_model.pkl"))

with open(os.path.join(BASE_DIR, "data/keypoint_classes.json"), "r") as f:
    class_names = json.load(f)
# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# STATE
# =========================
buffer = deque(maxlen=12)
wrist_history = deque(maxlen=8)

hand_present = False

last_time = time.time()
fps = 0

current_action = "NONE"

# =========================
# SWIPE DETECTION
# =========================
def detect_swipe():
    if len(wrist_history) < 6:
        return None

    diff = wrist_history[-1] - wrist_history[0]

    if diff > 0.18:
        return "Right_Swipe"
    elif diff < -0.18:
        return "Left_Swipe"

    return None

# =========================
# FEATURE EXTRACTION
# =========================
def extract(frame):
    global hand_present

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        hand_present = False
        return None

    hand_present = True
    lm = res.multi_hand_landmarks[0]

    wrist_history.append(lm.landmark[0].x)

    data = []
    for p in lm.landmark:
        data.extend([p.x, p.y, p.z])

    return np.array(data).reshape(1, -1)

# =========================
# MAP RAW MODEL OUTPUT → CLEAN LABELS
# =========================
def normalize(label):
    label = str(label).lower()

    if "left" in label:
        return "Left_Swipe"
    if "right" in label:
        return "Right_Swipe"
    if "stop" in label:
        return "Stop"
    if "thumb" in label and "up" in label:
        return "Thumbs_Up"
    if "thumb" in label and "down" in label:
        return "Thumbs_Down"

    return label

# =========================
# UI (UNCHANGED)
# =========================
HTML = """
<html>
<head>
<title>Jetson Gesture Control</title>
</head>
<body style="background:black;color:white;text-align:center;">
<h2>Gesture Control System</h2>
<img src="/video_feed" width="900">
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

# =========================
# STREAM
# =========================
def gen():
    global fps, last_time, buffer, current_action

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        features = extract(frame)

        gesture = "None"

        # =========================
        # RESET ON HAND LOSS
        # =========================
        if not hand_present:
            buffer.clear()
            wrist_history.clear()
            gesture = "None"
            current_action = "NONE"

        else:

            swipe = detect_swipe()

            if swipe:
                gesture = swipe
                buffer.append(gesture)
            else:
                if features is not None:
                    pred = normalize(model.predict(features)[0])
                    buffer.append(pred)
                else:
                    buffer.append("None")

                if len(buffer) > 0:
                    # majority vote smoothing
                    gesture = max(set(buffer), key=buffer.count)

        # =========================
        # MAP ACTION (DISPLAY ONLY)
        # =========================
        if gesture == "Thumbs_Up":
            current_action = "VOLUME UP"
        elif gesture == "Thumbs_Down":
            current_action = "VOLUME DOWN"
        elif gesture == "Stop":
            current_action = "PAUSE"
        elif gesture == "Left_Swipe":
            current_action = "SEEK BACK"
        elif gesture == "Right_Swipe":
            current_action = "SEEK FORWARD"
        else:
            current_action = "NONE"

        # =========================
        # FPS
        # =========================
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        # =========================
        # OVERLAY
        # =========================
        cv2.putText(frame, f"Gesture: {gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Action: {current_action}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Hand: {hand_present}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame, encode_param)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# =========================
# ROUTE
# =========================
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Stable Gesture Debug Build Running")
    app.run(host="0.0.0.0", port=5000, debug=False)