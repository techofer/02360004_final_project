import os
import shutil
import random
from pathlib import Path
from urllib.parse import quote
from argparse import ArgumentParser
import tqdm

def yatc_split(base_path, split_ratio=0.8, random_seed=42):
    # Configuration
    base_dir = Path(base_path)
    mix_dir = base_dir / "mix"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    # Ensure reproducibility
    random.seed(random_seed)

    # Ensure train and test directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # For each class folder in mix/
    for class_folder in tqdm.tqdm([x for x in mix_dir.iterdir()]):
        if class_folder.is_dir():
            images = list(class_folder.glob("*"))
            random.shuffle(images)
            
            split_idx = int(len(images) * split_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # Create class subfolders in train/ and test/
            sanitized_name = quote(class_folder.name, safe="")
            train_class_dir = train_dir / sanitized_name
            test_class_dir = test_dir / sanitized_name

            # Ensure it's not a file already
            if train_class_dir.exists() and not train_class_dir.is_dir():
                raise Exception(f"{train_class_dir} exists and is not a directory!")

            if test_class_dir.exists() and not test_class_dir.is_dir():
                raise Exception(f"{test_class_dir} exists and is not a directory!")

            train_class_dir.mkdir(parents=True, exist_ok=True)
            test_class_dir.mkdir(parents=True, exist_ok=True)
            # Move or copy images
            for img in train_images:
                shutil.copy(img, train_dir / class_folder.name / img.name)
            for img in test_images:
                shutil.copy(img, test_dir / class_folder.name / img.name)

    print("Dataset split complete with seed =", random_seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("base_path", help="Path to the base directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    args = parser.parse_args()
    yatc_split(args.base_path, args.split_ratio, args.seed)
