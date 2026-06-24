import os
import json
import random
import torch
import pandas as pd
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "gestures"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val")

BATCH_SIZE = 16
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------
# DATASET
# -----------------------
class GestureDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform

        # label column = index 1
        self.classes = sorted(self.data.iloc[:, 1].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        folder_name = row.iloc[0]
        label_str = row.iloc[1]

        folder_path = os.path.join(self.img_dir, folder_name)

        # -----------------------
        # SAFETY CHECK (IMPORTANT)
        # -----------------------
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        img_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if len(img_files) == 0:
            raise RuntimeError(f"No images in folder: {folder_path}")

        # -----------------------
        # SAMPLE RANDOM FRAME (IMPORTANT FIX)
        # -----------------------
        img_file = random.choice(img_files)
        img_path = os.path.join(folder_path, img_file)

        image = Image.open(img_path).convert("RGB")

        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------
# LOAD DATA
# -----------------------
train_data = GestureDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform)
val_data = GestureDataset(VAL_CSV, VAL_IMG_DIR, transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", class_names)

# save class mapping
with open("classes.json", "w") as f:
    json.dump(class_names, f)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# -----------------------
# LOSS / OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "gesture_model.pth")

print("Training complete.")
print("Saved: gesture_model.pth")
print("Saved: classes.json")