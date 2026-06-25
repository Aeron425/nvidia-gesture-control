import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD CLASSES
# -----------------------
with open("classes.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# -----------------------
# DATASET
# -----------------------
class GestureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.samples = []

        with open(csv_file, "r") as f:
            for line in f:
                folder, gesture, label = line.strip().split(";")
                self.samples.append((folder, int(label)))

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label = self.samples[idx]

        folder_path = os.path.join(self.root_dir, folder)

        images = sorted([
            x for x in os.listdir(folder_path)
            if x.endswith(".jpg")
        ])

        # use center frame
        img_path = os.path.join(folder_path, images[len(images)//2])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------
# TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------
# VALIDATION DATA
# -----------------------
val_dataset = GestureDataset(
    "gestures/val.csv",
    "gestures/val",
    transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(
    torch.load("gesture_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

# -----------------------
# VALIDATION
# -----------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nValidation Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")

