#!/usr/bin/env python3
"""
Run VGGT inference on a video or image folder and save predictions as .npz cache.
Used to generate data for the SelfEvo website demo.

Usage:
    # Pretrained (HuggingFace weights, no --ckpt_path)
    python run_inference.py --video_path examples/movie/zhenhuan/dance2.mov \
        --target_fps 10 --output exports/pretrained/zhenhuan-dance2/cache.npz

    # SelfEvo checkpoint
    python run_inference.py --video_path examples/movie/zhenhuan/dance2.mov \
        --target_fps 10 \
        --ckpt_path training/logs/zhenhuan_random_995_frcam/ckpts/checkpoint_11.pt \
        --output exports/selfevo/zhenhuan-dance2/cache.npz

    # Image folder input
    python run_inference.py --image_folder examples/movie/history/1 \
        --output exports/pretrained/history-1/cache.npz
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# Allow running from the repo root: PYTHONPATH=~/code/vggt python website/selfevo/tools/run_inference.py
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def sort_by_index(paths: List[str]) -> List[str]:
    def key(p):
        m = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
        return int(m.group(1)) if m else float("inf")
    return sorted(paths, key=key)


def _strip_module_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def load_model(ckpt_path: Optional[str], device: str) -> torch.nn.Module:
    model = VGGT()
    if ckpt_path is not None:
        print(f"[load] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = None
        for ema_key in ["ema_model", "ema_teacher"]:
            sd = ckpt.get(ema_key, None)
            if isinstance(sd, dict) and len(sd) > 0:
                state_dict = _strip_module_prefix(sd)
                print(f"[load] using {ema_key} weights")
                break
        if state_dict is None:
            state_dict = ckpt.get("model", ckpt)
            state_dict = _strip_module_prefix(state_dict)
            print("[load] EMA not found; using student weights")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[load] loading official facebook/VGGT-1B weights from HuggingFace")
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def extract_frames(video_path: str, out_dir: str, target_fps: float) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    existing = sort_by_index(glob.glob(os.path.join(out_dir, "*.png")))
    if existing:
        print(f"[video] reusing {len(existing)} existing frames in {out_dir}")
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    print(f"[video] source_fps={video_fps:.1f}  target_fps={target_fps:.1f}")

    paths, frame_id, saved_id, next_t = [], 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_id / video_fps
        if t + 1e-9 >= next_t:
            p = os.path.join(out_dir, f"{saved_id:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
            saved_id += 1
            next_t += 1.0 / target_fps
        frame_id += 1
    cap.release()
    print(f"[video] extracted {len(paths)} frames → {out_dir}")
    return paths


def run_inference(model: torch.nn.Module, image_names: List[str], device: str) -> dict:
    print(f"[infer] preprocessing {len(image_names)} images...")
    images = load_and_preprocess_images(image_names).to(device)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"[infer] running VGGT (dtype={dtype})...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images

    out = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr.squeeze(0)  # remove batch dim
            out[k] = arr
    print(f"[infer] done. keys={list(out.keys())}")
    return out


def main():
    parser = argparse.ArgumentParser(description="VGGT inference → .npz cache")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="Input video file (.mp4/.mov)")
    group.add_argument("--image_folder", type=str, help="Folder of pre-extracted frames")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="SelfEvo checkpoint .pt (omit to use official HuggingFace weights)")
    parser.add_argument("--target_fps", type=float, default=10.0,
                        help="Target FPS for frame sampling (video input only)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path, e.g. exports/pretrained/zhenhuan-dance2/cache.npz")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}")

    if args.video_path:
        stem = os.path.splitext(os.path.basename(args.video_path))[0]
        frames_dir = os.path.join(os.path.dirname(args.video_path), f"{stem}_frames")
        image_names = extract_frames(args.video_path, frames_dir, args.target_fps)
    else:
        image_names = sort_by_index(
            glob.glob(os.path.join(args.image_folder, "*.png")) +
            glob.glob(os.path.join(args.image_folder, "*.jpg")) +
            glob.glob(os.path.join(args.image_folder, "*.jpeg"))
        )
        print(f"[main] found {len(image_names)} images in {args.image_folder}")

    if not image_names:
        raise ValueError("No images found. Check --video_path / --image_folder.")

    model = load_model(args.ckpt_path, device)
    pred = run_inference(model, image_names, device)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.output, **pred)
    print(f"[main] saved → {args.output}  ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
