#!/usr/bin/env python3
"""
Merge a sparse inference npz (N frames) with a full set of input images (M frames).

For frames that were inferred: use real depth, extrinsic, intrinsic.
For frames that were NOT inferred: zero depth, linearly interpolated pose/intrinsic.

Usage:
    python merge_input_frames.py \
        --npz exports/pretrained/wild-dive/cache.npz \
        --full_images examples/dive \
        --inferred_indices 0 4 8 12 16 \
        --output exports/pretrained/wild-dive/cache_merged.npz
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np


def sort_by_index(paths):
    def key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)


def load_images(folder):
    """Load all images from folder as (M, 3, H, W) float32 array."""
    paths = sort_by_index(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        imgs.append(img)
    return np.stack(imgs, axis=0)  # (M, 3, H, W)


def interpolate_arrays(arr, inferred_indices, total):
    """Interpolate array values for missing indices using nearest inferred neighbors."""
    out = np.zeros((total,) + arr.shape[1:], dtype=arr.dtype)
    for j, idx in enumerate(inferred_indices):
        out[idx] = arr[j]

    # Linear interpolation between inferred frames
    for i in range(total):
        if i in inferred_indices:
            continue
        # Find nearest left and right inferred indices
        left = max([idx for idx in inferred_indices if idx <= i], default=None)
        right = min([idx for idx in inferred_indices if idx >= i], default=None)
        if left is None:
            out[i] = out[right]
        elif right is None:
            out[i] = out[left]
        elif left == right:
            out[i] = out[left]
        else:
            t = (i - left) / (right - left)
            out[i] = (1 - t) * out[left] + t * out[right]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Sparse inference npz (N frames)")
    parser.add_argument("--full_images", required=True, help="Folder with all M input images")
    parser.add_argument("--inferred_indices", type=int, nargs="+", required=True,
                        help="Indices (0-based) into the full image list that were inferred")
    parser.add_argument("--output", required=True, help="Output merged npz path")
    args = parser.parse_args()

    cache = np.load(args.npz, allow_pickle=True)
    inferred = sorted(args.inferred_indices)
    N = len(inferred)

    # Load sparse inference results
    depth = cache["depth"]          # (N, H, W) or (N, H, W, 1)
    extrinsic = cache["extrinsic"]  # (N, 3, 4)
    intrinsic = cache["intrinsic"]  # (N, 3, 3)

    # Load all input images
    all_images = load_images(args.full_images)  # (M, 3, H, W)
    M = all_images.shape[0]
    print(f"  Inferred frames: {N}, Total input images: {M}")
    print(f"  Inferred indices: {inferred}")

    # Interpolate extrinsic and intrinsic for missing frames
    merged_extrinsic = interpolate_arrays(extrinsic, inferred, M)
    merged_intrinsic = interpolate_arrays(intrinsic, inferred, M)

    # Depth: zero for non-inferred frames
    if depth.ndim == 4:
        merged_depth = np.zeros((M,) + depth.shape[1:], dtype=depth.dtype)
    else:
        merged_depth = np.zeros((M,) + depth.shape[1:], dtype=depth.dtype)
    for j, idx in enumerate(inferred):
        merged_depth[idx] = depth[j]

    # Confidence: propagate if present
    save_dict = {
        "images": all_images,
        "depth": merged_depth,
        "extrinsic": merged_extrinsic,
        "intrinsic": merged_intrinsic,
    }
    for conf_key in ("depth_conf", "world_points_conf"):
        if conf_key in cache:
            conf = cache[conf_key]
            merged_conf = np.zeros((M,) + conf.shape[1:], dtype=conf.dtype)
            for j, idx in enumerate(inferred):
                merged_conf[idx] = conf[j]
            save_dict[conf_key] = merged_conf

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, **save_dict)
    print(f"  Saved merged npz: {args.output} ({M} frames)")


if __name__ == "__main__":
    main()
