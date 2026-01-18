import os
import shutil
import random
from tqdm import tqdm

DATASET_DIR = "data/train"
OUTPUT_DIR = "data"

CLASSES = ["ai_art", "human_art"]

VAL_SPLIT = 0.2   # increase val size
TEST_SPLIT = 0.2  # increase test size

def make_dirs():
    for split in ["val", "test"]:
        for cls in CLASSES:
            path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

def split_data():
    for cls in CLASSES:
        class_dir = os.path.join(DATASET_DIR, cls)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'))]

        random.shuffle(images)

        total = len(images)
        val_count = int(total * VAL_SPLIT)
        test_count = int(total * TEST_SPLIT)

        val_imgs = images[:val_count]
        test_imgs = images[val_count:val_count+test_count]

        print(f"\nFound {total} images in '{cls}'.")
        print(f"→ Moving {len(val_imgs)} to val")
        print(f"→ Moving {len(test_imgs)} to test")

        for img_list, split_name in [(val_imgs, "val"), (test_imgs, "test")]:
            for img in tqdm(img_list, desc=f"{cls} → {split_name}"):
                src = os.path.join(class_dir, img)
                dst = os.path.join(OUTPUT_DIR, split_name, cls, img)
                shutil.move(src, dst)   # FIXED

if __name__ == "__main__":
    make_dirs()
    split_data()
    print("\n✅ Dataset split completed successfully!")
