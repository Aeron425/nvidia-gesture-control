import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import time
from flask import Flask, Response, render_template_string
from collections import deque

# =========================
# APP SETUP
# =========================
app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("gesture_keypoint_model.pkl")

with open("keypoint_classes.json", "r") as f:
    class_names = json.load(f)

# =========================
# CAMERA (OPTIMIZED)
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
buffer = deque(maxlen=10)
wrist_history = deque(maxlen=8)

last_time = time.time()
fps = 0

current_action = "NONE"
last_action_time = 0
ACTION_COOLDOWN = 1.0

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0]

    wrist_history.append(lm.landmark[0].x)

    row = []
    for p in lm.landmark:
        row.extend([p.x, p.y, p.z])

    return np.array(row).reshape(1, -1)

# =========================
# ACTION MAP
# =========================
def trigger_action(gesture):
    global last_action_time, current_action

    now = time.time()

    if now - last_action_time < ACTION_COOLDOWN:
        return None

    action = None

    if gesture == "Thumbs_Up":
        action = "VOLUME UP"
    elif gesture == "Thumbs_Down":
        action = "VOLUME DOWN"
    elif gesture == "Stop":
        action = "PAUSE"
    elif gesture == "Left_Swipe":
        action = "SEEK BACK"
    elif gesture == "Right_Swipe":
        action = "SEEK FORWARD"

    if action:
        last_action_time = now
        current_action = action

    return action

# =========================
# UI
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
# STREAM GENERATOR
# =========================
def gen():
    global fps, last_time, buffer

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # reduce lag
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        # optional frame skip (uncomment if lag)
        # cap.grab()

        features = extract(frame)

        gesture = "None"

        # -----------------------
        # SWIPE PRIORITY (motion)
        # -----------------------
        swipe = detect_swipe()
        if swipe:
            gesture = swipe
            buffer.append(gesture)
        else:
            if features is not None:
                pred = model.predict(features)[0]
                buffer.append(pred)
            else:
                buffer.append("None")

            try:
                gesture = max(set(buffer), key=buffer.count)
            except:
                gesture = "None"

        trigger_action(gesture)

        # -----------------------
        # FPS CALC
        # -----------------------
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        # -----------------------
        # OVERLAY
        # -----------------------
        cv2.putText(frame,
                    f"Gesture: {gesture}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Action: {current_action}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2)

        _, jpeg = cv2.imencode('.jpg', frame, encode_param)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# =========================
# STREAM ROUTE
# =========================
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)