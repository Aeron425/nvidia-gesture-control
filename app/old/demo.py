import cv2
import torch
import json
import numpy as np
from collections import deque
from torchvision import transforms, models
import torch.nn.functional as F

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

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
# SMOOTHING BUFFER
# -----------------------
buffer = deque(maxlen=12)

# confidence threshold
CONF_THRESH = 0.70

# -----------------------
# CAMERA
# -----------------------
cap = cv2.VideoCapture(0)

print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip for natural view
    frame = cv2.flip(frame, 1)

    img = transform(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

        conf = conf.item()
        pred_class = class_names[pred.item()]

    # -----------------------
    # FILTER LOW CONFIDENCE
    # -----------------------
    if conf > CONF_THRESH:
        buffer.append(pred_class)
    else:
        buffer.append("none")

    # -----------------------
    # STABLE PREDICTION
    # -----------------------
    try:
        final_pred = max(set(buffer), key=buffer.count)
    except:
        final_pred = "none"

    # -----------------------
    # DISPLAY
    # -----------------------
    cv2.putText(frame, f"Gesture: {final_pred}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Conf: {conf:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Gesture Control", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()