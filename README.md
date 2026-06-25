# 🧠 Jetson Gesture Control System

Real-time AI hand gesture recognition system running on NVIDIA Jetson using MediaPipe + OpenCV + a trained ML model.

---

## ⚡ Features

- Real-time hand tracking (MediaPipe)
- ML-based gesture classification
- Swipe detection (left / right)
- Stable prediction smoothing (buffer voting)
- Auto reset when hand leaves frame
- Live browser video stream (Flask MJPEG)
- On-screen overlay (gesture, action, FPS)

---

## ✋ Supported Gestures

- Thumbs Up → Volume Up (display only)
- Thumbs Down → Volume Down (display only)
- Stop → Pause
- Left Swipe → Seek Back
- Right Swipe → Seek Forward

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
cd app
python3 main.py

http://<JETSON_IP>:5000

🧠 Tech Stack
Python
OpenCV
MediaPipe
Flask
scikit-learn
NumPy

⚙️ Hardware
NVIDIA Jetson Orin Nano
USB / CSI Camera
Linux (JetPack)