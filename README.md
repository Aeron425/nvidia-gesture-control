# NVIDIA Gesture Control (Jetson AI Project)

Real-time hand gesture recognition system using MediaPipe + machine learning, running on NVIDIA Jetson.  
It detects hand gestures through a camera stream and maps them to actions like volume control, pause, and media navigation.

---

## The Algorithm

This project uses a hybrid approach:

### 1. Hand Tracking (MediaPipe)
- Uses MediaPipe Hands to detect 21 hand landmarks in real-time
- Each frame extracts (x, y, z) coordinates of landmarks

### 2. Gesture Classification (ML Model)
- A trained machine learning model (`gesture_keypoint_model.pkl`) predicts gestures
- Input: flattened 21-point hand landmark vector
- Output: gesture label (e.g. Thumbs_Up, Left_Swipe)

### 3. Swipe Detection (Temporal Logic)
- Tracks wrist movement over time
- Compares horizontal displacement to detect:
  - Left Swipe
  - Right Swipe

### 4. Action Mapping
Detected gestures are mapped to UI actions:
- Thumbs Up → Volume Up
- Thumbs Down → Volume Down
- Stop → Pause
- Left Swipe → Seek Back
- Right Swipe → Seek Forward

### 5. Web Interface
- Flask server streams live camera feed
- Displays:
  - Detected gesture
  - Current action
  - FPS counter
 
Video Link:

[video][(https://youtu.be/_4fd6zi6N4M](url))

---

## Dependencies and Usage

Install required libraries:

```bash
pip install opencv-python mediapipe numpy scikit-learn flask joblib

Clone Repo:

git clone https://github.com/YOUR_USERNAME/nvidia-gesture-control.git
cd nvidia-gesture-control

Run Application & Open Link:

python3 app/main.py
http://localhost(replace):5000





