# Jetson Gesture Control

Real-time hand gesture recognition system using MediaPipe + OpenCV + ML.

---

## Run

```bash
pip install -r requirements.txt
cd app
python3 main.py

**## Open**
http://<JETSON_IP>:5000

**Features**
Hand tracking
Gesture classification
Swipe detection
Live video stream
Auto reset on hand loss
On-screen overlay (gesture + action + FPS)

**Gestures**
Thumbs Up → Volume Up
Thumbs Down → Volume Down
Stop → Pause
Left Swipe → Seek Back
Right Swipe → Seek Forward

**What you need**
