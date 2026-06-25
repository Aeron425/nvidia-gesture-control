import os
import shutil

DATASET_ROOT = "gestures"
OUTPUT_ROOT = "gestures_clean"

gesture_map = {
    "Left Swipe_new": "left",
    "Left_Swipe_new": "left",

    "Right Swipe_new": "right",
    "Right_Swipe_new": "right",

    "Stop Gesture_new": "stop",
    "Stop_new": "stop",

    "Thumbs Up_new": "thumb_up",
    "Thumbs_Up_new": "thumb_up",

    "Thumbs Down_new": "thumb_down",
    "Thumbs_Down_new": "thumb_down",
}

splits = ["train", "val"]

for split in splits:
    split_path = os.path.join(DATASET_ROOT, split)
    csv_file = os.path.join(DATASET_ROOT, f"{split}.csv")

    print(f"Processing {split}...")

    with open(csv_file, "r") as f:
        for line in f:
            folder, gesture, label = line.strip().split(";")

            if gesture not in gesture_map:
                continue

            new_label = gesture_map[gesture]

            src_folder = os.path.join(split_path, folder)

            if not os.path.isdir(src_folder):
                continue

            dst_folder = os.path.join(OUTPUT_ROOT, split, new_label)
            os.makedirs(dst_folder, exist_ok=True)

            # copy frames
            for img in os.listdir(src_folder):
                if img.endswith(".jpg") or img.endswith(".png"):
                    src = os.path.join(src_folder, img)
                    dst = os.path.join(dst_folder, f"{folder}_{img}")
                    shutil.copy2(src, dst)

print("Dataset cleaned and ready!")
