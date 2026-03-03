#!/usr/bin/env python3
"""
Prepare web assets from the standard dataset.

Converts the Python pipeline outputs (CSV stats, original JPEGs) into
web-ready assets (dataset.json + resized images) for GitHub Pages hosting.

This is a one-time preprocessing step that packages the standard dataset
so anyone can play without needing to run SAM or any Python code.
"""

import json
import csv
from pathlib import Path
from PIL import Image
from collections import defaultdict

def extract_image_number(path_str):
    """Extract image number from path like 'image_10.jpg' -> 10."""
    return int(Path(path_str).stem.split('_')[1])

def prepare_web_assets(
    stats_csv: Path,
    feature_map_json: Path,
    converted_images_dir: Path,
    bingo_cards_json: Path,
    output_dir: Path,
):
    """
    Build web assets from pipeline outputs.

    Reads segmentation_stats.csv and image_feature_map.json,
    generates dataset.json with per-image feature counts,
    resizes images to web format, and copies bingo_cards.json.
    """

    # Create output directories
    data_dir = output_dir / "data"
    images_dir = output_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load feature map to know all features and images
    print("Loading image_feature_map.json...")
    with open(feature_map_json) as f:
        feature_map = json.load(f)

    # Get all unique feature names from the map
    all_features = set()
    image_paths = {}  # maps basename like "image_1.jpg" -> absolute path
    for abs_path, features in feature_map.items():
        basename = Path(abs_path).name
        image_paths[basename] = abs_path
        all_features.update(features)

    all_features = sorted(all_features)
    print(f"  Found {len(all_features)} unique features")
    print(f"  Found {len(image_paths)} images")

    # Step 2: Load CSV and build n_objects per (image, feature) pair
    print("Loading segmentation_stats.csv...")
    counts = defaultdict(lambda: defaultdict(int))

    with open(stats_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image']
            feature = row['feature']
            n_objects = int(row['n_objects']) if row['n_objects'] else 0

            basename = Path(image_path).name
            counts[basename][feature] = n_objects

    print(f"  Processed {len(counts)} images from CSV")

    # Step 3: Build dataset.json with all images in order
    print("Building dataset.json...")
    images_list = []

    # Sort by image number to ensure consistent order
    sorted_basenames = sorted(
        image_paths.keys(),
        key=lambda x: extract_image_number(x)
    )

    for basename in sorted_basenames:
        # Get all features for this image (from feature_map)
        abs_path = image_paths[basename]
        labeled_features = feature_map[abs_path]

        # Build feature counts dict: feature -> n_objects
        features_dict = {}
        for feat in all_features:
            # Use count from CSV if it exists, otherwise 0
            features_dict[feat] = counts[basename].get(feat, 0)

        image_entry = {
            "id": basename.replace('.jpg', ''),  # e.g., "image_1"
            "filename": f"images/{basename}",
            "features": features_dict
        }
        images_list.append(image_entry)

    dataset = {
        "version": 1,
        "images": images_list
    }

    dataset_path = data_dir / "dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"  Wrote {dataset_path}")

    # Step 4: Resize images
    print("Resizing and copying images...")
    for i, basename in enumerate(sorted_basenames, 1):
        src_path = converted_images_dir / basename
        dst_path = images_dir / basename

        if src_path.exists():
            img = Image.open(src_path)
            # Resize to max 800x600, maintaining aspect ratio
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            # Save as JPEG with quality 75
            img.save(dst_path, 'JPEG', quality=75)
            print(f"  [{i}/{len(sorted_basenames)}] {basename}")
        else:
            print(f"  WARNING: {src_path} not found, skipping")

    # Step 5: Copy bingo_cards.json
    print("Copying bingo_cards.json...")
    cards_dst = data_dir / "cards.json"
    with open(bingo_cards_json) as f:
        cards_data = json.load(f)
    with open(cards_dst, 'w') as f:
        json.dump(cards_data, f, indent=2)
    print(f"  Wrote {cards_dst} ({len(cards_data)} cards)")

    print("\n✓ Web assets prepared successfully!")
    print(f"  Total: {len(images_list)} images, {len(all_features)} features")
    print(f"  Output directory: {output_dir}")

if __name__ == '__main__':
    # Default paths relative to project root
    project_root = Path(__file__).parent.parent

    prepare_web_assets(
        stats_csv=project_root / "mask" / "segmentation_stats.csv",
        feature_map_json=project_root / "image_feature_map.json",
        converted_images_dir=project_root / "converted_images",
        bingo_cards_json=project_root / "bingo_cards.json",
        output_dir=project_root / "web",
    )
