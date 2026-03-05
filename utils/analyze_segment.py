#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from transformers import Sam3Model, Sam3Processor
import numpy as np
import pandas as pd
import rasterio
import torch
import sys
import os
import random


def safe_feature_name(name: str) -> str:
    """Converts a feature name to a filesystem-safe string.

    Args:
        name: The raw feature name.

    Returns:
        The name with non-alphanumeric characters (except hyphens and underscores)
        replaced by underscores, with leading/trailing underscores stripped.
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


def open_image_pil(path):
    """Opens an image file as an RGBA PIL Image.

    Args:
        path: Path to the image file.

    Returns:
        PIL.Image.Image: The loaded image in RGBA mode.
    """
    return Image.open(path).convert("RGBA")


def overlay_mask_on_image(img_pil: Image.Image, mask_arr: np.ndarray, color=(255, 0, 0), alpha=0.5):
    """Composites a coloured mask overlay onto a base image.

    Args:
        img_pil: Base image in RGBA mode.
        mask_arr: 2-D numpy array where non-zero pixels indicate the mask.
        color: RGB tuple for the overlay colour.
        alpha: Opacity of the overlay in [0, 1].

    Returns:
        PIL.Image.Image: The composited RGBA image.
    """
    H, W = mask_arr.shape
    if img_pil.width != W or img_pil.height != H:
        img_pil = img_pil.resize((W, H), Image.LANCZOS)

    color_with_alpha = (color[0], color[1], color[2], int(255 * alpha))
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    mask_bool = mask_arr > 0
    rgba[mask_bool, 0] = color_with_alpha[0]
    rgba[mask_bool, 1] = color_with_alpha[1]
    rgba[mask_bool, 2] = color_with_alpha[2]
    rgba[mask_bool, 3] = color_with_alpha[3]

    mask_img = Image.fromarray(rgba, mode="RGBA")
    overlay = Image.alpha_composite(img_pil, mask_img)
    return overlay


def compute_stats_from_files(mask_path: Path, scores_path: Path):
    """Reads saved mask and score GeoTIFFs and computes per-feature statistics.

    Args:
        mask_path: Path to the unique-ID mask GeoTIFF.
        scores_path: Path to the confidence score GeoTIFF.

    Returns:
        dict: Keys are n_objects, mask_pixels, coverage_pct, mean_score,
              coverage_area_m2. Any value may be None if the file is unreadable.
    """
    n_objects = None
    mask_pixels = None
    coverage_pct = None
    mean_score = None
    coverage_area_m2 = None
    arr = None
    try:
        with rasterio.open(str(mask_path)) as src:
            arr = src.read(1)
            H, W = arr.shape
            mask_pixels = int((arr > 0).sum())
            total_pixels = int(H * W)
            coverage_pct = 100.0 * mask_pixels / total_pixels if total_pixels > 0 else 0.0
            unique_vals = np.unique(arr)
            n_objects = int(len([v for v in unique_vals if int(v) != 0]))
            try:
                tr = src.transform
                pixel_area = abs(tr.a * tr.e - tr.b * tr.d)
                coverage_area_m2 = float(pixel_area * mask_pixels)
            except Exception:
                coverage_area_m2 = None
    except Exception as e:
        print("Warning: failed to read mask file:", mask_path, e)

    try:
        with rasterio.open(str(scores_path)) as ss:
            scores = ss.read(1)
            if mask_pixels and mask_pixels > 0 and arr is not None:
                mean_score = float(scores[arr > 0].mean())
            else:
                mean_score = float(scores.mean())
    except Exception as e:
        print("Warning: failed to read scores file:", scores_path, e)

    return dict(
        n_objects=n_objects,
        mask_pixels=mask_pixels,
        coverage_pct=coverage_pct,
        mean_score=mean_score,
        coverage_area_m2=coverage_area_m2,
    )


def save_masks_to_tif(masks, scores, source_path, mask_out, scores_out):
    """Writes unique-ID mask and per-pixel score GeoTIFFs using rasterio.

    Each detected object receives a unique integer ID in the mask raster
    (1, 2, 3, …); background pixels are 0. The scores raster stores the
    confidence value of whichever mask covers each pixel.

    Args:
        masks: List of (H, W) boolean numpy arrays, one per detected object.
        scores: List of float confidence scores, one per mask.
        source_path: Source image path; used to inherit CRS and geotransform
            when the source is a GeoTIFF.
        mask_out: Output path for the mask GeoTIFF.
        scores_out: Output path for the scores GeoTIFF.

    Raises:
        ValueError: If masks is empty.
    """
    if len(masks) == 0:
        raise ValueError("No masks to save")

    H, W = masks[0].shape
    mask_array = np.zeros((H, W), dtype=np.uint32)
    score_array = np.zeros((H, W), dtype=np.float32)

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        mask_bool = mask > 0
        mask_array[mask_bool] = i
        score_array[mask_bool] = float(score)

    try:
        with rasterio.open(str(source_path)) as src:
            crs = src.crs
            transform = src.transform
    except Exception:
        crs = None
        transform = rasterio.transform.from_bounds(0, 0, W, H, W, H)

    meta = dict(driver="GTiff", height=H, width=W, count=1,
                compress="deflate", crs=crs, transform=transform)

    with rasterio.open(str(mask_out), "w", dtype=np.uint8, **meta) as dst:
        dst.write(mask_array.clip(0, 255).astype(np.uint8), 1)

    with rasterio.open(str(scores_out), "w", dtype=np.float32, **meta) as dst:
        dst.write(score_array, 1)


def process(mapping_path: Path, out_dir: Path, device: str = "gpu", resume=True, overlay_alpha=0.45, hf_token=None):
    """Runs SAM3 segmentation on all image–feature pairs in the mapping file.

    For each image, loads it once into the model then iterates over its
    assigned features, generating and saving a mask + score GeoTIFF per
    feature. Optionally resumes from previously saved files.

    Args:
        mapping_path: Path to image_feature_map.json mapping image paths to
            lists of feature strings.
        out_dir: Output directory; masks/ and overlays/ subdirectories are
            created automatically.
        device: "gpu" to use CUDA or MPS if available, "cpu" to force CPU.
        resume: If True, skip image–feature pairs whose output files already
            exist.
        overlay_alpha: Opacity for mask overlay PNGs in [0, 1].
        hf_token: Optional HuggingFace access token for gated models.

    Returns:
        pd.DataFrame: One row per image–feature pair with columns image,
            feature, mask_file, scores_file, overlay_file, n_objects,
            mask_pixels, coverage_pct, coverage_area_m2, mean_score.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_path}")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    out_dir = Path(out_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    if device == "gpu":
        if torch.cuda.is_available():
            device_idx = "cuda"
        elif torch.backends.mps.is_available():
            device_idx = "mps"
            print("Using Apple Silicon GPU (MPS)")
        else:
            device_idx = "cpu"
            print("GPU requested but not available. Using CPU.")
    else:
        device_idx = "cpu"

    MODEL_ID = "facebook/sam3"
    print("Loading Sam3Model from HuggingFace, device:", device_idx)
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to(device_idx)
    model.eval()

    stats_rows = []
    color_cache = {}

    for img_str, features in mapping.items():
        if not features:
            print(f"Skipping {img_str}: no features selected.")
            continue
        img_path = Path(img_str)
        if not img_path.exists():
            print(f"Skipping missing image: {img_path}")
            continue
        print(f"Processing image: {img_path}, features: {features}")

        pil_image = Image.open(str(img_path)).convert("RGB")

        fname_base = img_path.stem

        try:
            img_pil = open_image_pil(img_path)
        except Exception:
            img_pil = None

        for feat in features:
            feat_safe = safe_feature_name(feat)
            mask_out = masks_dir / f"{fname_base}__{feat_safe}_masks.tif"
            scores_out = masks_dir / f"{fname_base}__{feat_safe}_scores.tif"
            overlay_out = overlays_dir / f"{fname_base}__{feat_safe}_overlay.png"

            if resume and mask_out.exists() and scores_out.exists():
                print("  Skipping", feat, "(already exists)")
                stats = compute_stats_from_files(mask_out, scores_out)
                stats_rows.append({
                    "image": str(img_path),
                    "feature": feat,
                    "mask_file": str(mask_out),
                    "scores_file": str(scores_out),
                    "overlay_file": str(overlay_out) if img_pil is not None else None,
                    "n_objects": stats["n_objects"],
                    "mask_pixels": stats["mask_pixels"],
                    "coverage_pct": stats["coverage_pct"],
                    "coverage_area_m2": stats["coverage_area_m2"],
                    "mean_score": stats["mean_score"],
                })
                continue

            print("  Generating masks for feature:", feat)
            try:
                inputs = processor(images=pil_image, text=feat, return_tensors="pt").to(device_idx)
                with torch.no_grad():
                    outputs = model(**inputs)
                H, W = pil_image.height, pil_image.width
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=[[H, W]],
                )[0]
                masks = [m.cpu().numpy() for m in results["masks"]]
                scores = [float(s) for s in results["scores"]]
                print(f"  Found {len(masks)} objects.")
            except Exception as e:
                print("  Error during segmentation:", e)
                stats_rows.append({
                    "image": str(img_path),
                    "feature": feat,
                    "mask_file": None,
                    "scores_file": None,
                    "overlay_file": None,
                    "n_objects": 0,
                    "mask_pixels": 0,
                    "coverage_pct": 0.0,
                    "coverage_area_m2": None,
                    "mean_score": None,
                })
                continue

            try:
                save_masks_to_tif(masks, scores, img_path, mask_out, scores_out)
                print("  Saved mask:", mask_out, "scores:", scores_out)
            except ValueError as e:
                print("  save skipped:", e)
                stats_rows.append({
                    "image": str(img_path),
                    "feature": feat,
                    "mask_file": None,
                    "scores_file": None,
                    "overlay_file": None,
                    "n_objects": 0,
                    "mask_pixels": 0,
                    "coverage_pct": 0.0,
                    "coverage_area_m2": None,
                    "mean_score": None,
                })
                continue
            except Exception as e:
                print("  Unexpected error saving masks:", e)
                stats_rows.append({
                    "image": str(img_path),
                    "feature": feat,
                    "mask_file": None,
                    "scores_file": None,
                    "overlay_file": None,
                    "n_objects": 0,
                    "mask_pixels": 0,
                    "coverage_pct": 0.0,
                    "coverage_area_m2": None,
                    "mean_score": None,
                })
                continue

            stats = compute_stats_from_files(mask_out, scores_out)
            palette_color = color_cache.get(feat)
            if palette_color is None:
                palette_color = tuple([int(x) for x in np.random.RandomState(abs(hash(feat)) % (2**32)).randint(50, 255, size=3)])
                color_cache[feat] = palette_color

            if img_pil is not None and mask_out.exists():
                try:
                    with rasterio.open(str(mask_out)) as msrc:
                        mask_arr = msrc.read(1)
                        overlay_img = overlay_mask_on_image(img_pil, mask_arr, color=palette_color, alpha=overlay_alpha)
                        overlay_img.save(overlay_out)
                except Exception as e:
                    print("  Warning: failed to create overlay:", e)

            stats_rows.append({
                "image": str(img_path),
                "feature": feat,
                "mask_file": str(mask_out) if mask_out.exists() else None,
                "scores_file": str(scores_out) if scores_out.exists() else None,
                "overlay_file": str(overlay_out) if (img_pil is not None and overlay_out.exists()) else None,
                "n_objects": stats["n_objects"],
                "mask_pixels": stats["mask_pixels"],
                "coverage_pct": stats["coverage_pct"],
                "coverage_area_m2": stats["coverage_area_m2"],
                "mean_score": stats["mean_score"],
            })

    df = pd.DataFrame(stats_rows)
    csv_out = out_dir / "segmentation_stats.csv"
    df.to_csv(csv_out, index=False)
    print("Saved stats CSV:", csv_out)
    return df


def parse_args():
    """Parses command-line arguments for the segmentation pipeline.

    Returns:
        argparse.Namespace: Parsed arguments with mapping, out, device,
            and resume attributes.
    """
    parser = argparse.ArgumentParser(description="Batch analyze SAM masks from mapping JSON")
    parser.add_argument("--mapping", "-m", default="image_feature_map.json", help="Path to mapping json")
    parser.add_argument("--out", "-o", default="mask", help="Output folder for masks, overlays and CSV")
    parser.add_argument("--device", "-d", choices=["gpu", "cpu"], default="gpu", help="Device to use")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Overwrite existing masks and scores")
    return parser.parse_args()


def main():
    """Entry point: resolves paths relative to the repo root and runs process()."""
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    mapping_arg = Path(args.mapping)
    if not mapping_arg.is_absolute():
        mapping_arg = repo_root / mapping_arg

    out_arg = Path(args.out)
    if not out_arg.is_absolute():
        out_arg = repo_root / out_arg

    df = process(mapping_path=mapping_arg, out_dir=out_arg, device=args.device, resume=args.resume)
    print("Done. Results rows:", len(df))


if __name__ == "__main__":
    main()
