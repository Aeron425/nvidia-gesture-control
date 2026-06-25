import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "gestures_clean"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------
# TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------
# DATASETS
# -----------------------
train_data = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=transform
)

val_data = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # IMPORTANT for Jetson stability
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", class_names)

# save classes
with open("classes.json", "w") as f:
    json.dump(class_names, f)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# -----------------------
# LOSS / OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    loop = tqdm(train_loader, leave=True)

    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)

    # -----------------------
    # VALIDATION
    # -----------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1} Done | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "gesture_model.pth")
print("\nSaved: gesture_model.pth")
print("Saved: classes.json")