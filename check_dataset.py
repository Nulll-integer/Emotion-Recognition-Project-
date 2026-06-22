# check_dataset.py
# Verifies your dataset is correctly organized

import os

dataset_path = "dataset"
splits = ["train", "test"]
emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

print("=" * 50)
print("DATASET VERIFICATION REPORT")
print("=" * 50)

total_images = 0
all_good = True

for split in splits:
    print(f"\n📁 {split.upper()} SET:")
    split_total = 0

    for emotion in emotions:
        folder = os.path.join(dataset_path, split, emotion)

        if not os.path.exists(folder):
            print(f"  ❌ MISSING: {emotion}/")
            all_good = False
        else:
            count = len([f for f in os.listdir(folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✅ {emotion:<12} → {count:,} images")
            split_total += count

    print(f"  📊 {split.upper()} TOTAL: {split_total:,} images")
    total_images += split_total

print("\n" + "=" * 50)
print(f"📊 GRAND TOTAL: {total_images:,} images")

if all_good:
    print("✅ Dataset structure is CORRECT — Ready for training!")
else:
    print("❌ Fix the missing folders above before continuing!")

print("=" * 50)