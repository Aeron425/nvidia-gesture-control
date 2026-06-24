import cv2
import torch
import time
import json
import numpy as np
from collections import deque
from torchvision import transforms, models

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load classes
with open("classes.json", "r") as f:
    class_names = json.load(f)

NUM_FRAMES = 30
buffer = deque(maxlen=NUM_FRAMES)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load("gesture_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# CAMERA
# -----------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: camera not found")
    exit()

print("Running... press Ctrl+C to stop")

# -----------------------
# LOOP
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    buffer.append(frame)

    # wait until buffer fills
    if len(buffer) < NUM_FRAMES:
        continue

    # run inference on all frames
    probs_sum = None

    with torch.no_grad():
        for f in buffer:
            img = transform(f).unsqueeze(0).to(DEVICE)
            out = model(img)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs

    probs_avg = probs_sum / NUM_FRAMES

    pred_idx = np.argmax(probs_avg)
    confidence = probs_avg[pred_idx]

    label = class_names[pred_idx]

    print(f"Gesture: {label} | Confidence: {confidence:.2f}")

    # optional debug view (SSH-safe)
    cv2.imwrite("latest_frame.jpg", frame)
