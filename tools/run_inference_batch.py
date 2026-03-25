#!/usr/bin/env python3
"""
Batch inference: load the model once, run on many image-folder scenes or video files.

Image-folder mode (e.g. DAVIS):
    python run_inference_batch.py \
        --input_root examples/DAVIS_480p_1of5 \
        --output_root website/selfevo/exports/pretrained \
        --prefix davis- --mask_sky

Video-folder mode (e.g. wild, wild2):
    python run_inference_batch.py \
        --video_dir examples/wild \
        --output_root website/selfevo/exports/pretrained \
        --prefix wild- --target_fps 1 --mask_sky

With SelfEvo checkpoint:
    python run_inference_batch.py ... \
        --ckpt_path training/logs/random_995_frcam/ckpts/checkpoint_11.pt
"""
import argparse
import glob
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def sort_by_index(paths: List[str]) -> List[str]:
    def key(p):
        m = re.search(r"(\d+)", os.path.splitext(os.path.basename(p))[0])
        return int(m.group(1)) if m else float("inf")
    return sorted(paths, key=key)


def _strip_module_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()}


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def extract_frames_from_video(video_path: str, out_dir: str,
                               target_fps: float, overwrite: bool = False) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    existing = sort_by_index(
        glob.glob(os.path.join(out_dir, "*.jpg")) +
        glob.glob(os.path.join(out_dir, "*.png"))
    )
    if existing and not overwrite:
        print(f"  [video] reusing {len(existing)} existing frames in {out_dir}")
        return existing

    source_fps = get_video_fps(video_path)
    print(f"  [video] source_fps={source_fps:.1f}  target_fps={target_fps:.1f}")

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    saved_id = 0
    frame_id = 0
    next_sample_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_id / source_fps
        if current_time >= next_sample_time:
            out_path = os.path.join(out_dir, f"{saved_id:05d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_paths.append(out_path)
            saved_id += 1
            next_sample_time += 1.0 / target_fps
        frame_id += 1
    cap.release()
    print(f"  [video] extracted {len(frame_paths)} frames → {out_dir}")
    return frame_paths


# ─────────────────────────────────────────────
# Sky segmentation (mirrors demo_viser.py logic)
# ─────────────────────────────────────────────

SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"

def _ensure_skyseg_onnx(onnx_path: str):
    if not os.path.exists(onnx_path):
        print("[sky] Downloading skyseg.onnx ...")
        from visual_util import download_file_from_url
        download_file_from_url(SKYSEG_URL, onnx_path)


def apply_sky_segmentation(conf: np.ndarray, image_folder: str,
                            onnx_path: str = "skyseg.onnx") -> np.ndarray:
    if onnxruntime is None:
        print("  [sky] WARNING: onnxruntime not installed, skipping sky masking")
        return conf

    _ensure_skyseg_onnx(onnx_path)
    from visual_util import segment_sky

    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    session = onnxruntime.InferenceSession(onnx_path)
    image_files = sort_by_index(
        glob.glob(os.path.join(image_folder, "*.jpg")) +
        glob.glob(os.path.join(image_folder, "*.jpeg")) +
        glob.glob(os.path.join(image_folder, "*.png"))
    )

    sky_mask_list = []
    for image_path in image_files[:S]:
        mask_path = os.path.join(sky_masks_dir, os.path.basename(image_path))
        if os.path.exists(mask_path):
            sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, session, mask_path)
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        sky_mask_list.append(sky_mask)

    sky_arr = np.array(sky_mask_list, dtype=np.float32)
    conf = conf * (sky_arr > 0.1).astype(np.float32)
    print(f"  [sky] sky segmentation applied ({S} frames)")
    return conf


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def load_model(ckpt_path: Optional[str], device: str) -> torch.nn.Module:
    model = VGGT()
    if ckpt_path is not None:
        print(f"[load] checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = None
        for key in ("ema_model", "ema_teacher"):
            sd = ckpt.get(key)
            if isinstance(sd, dict) and len(sd) > 0:
                state_dict = _strip_module_prefix(sd)
                print(f"[load] using {key} weights")
                break
        if state_dict is None:
            state_dict = _strip_module_prefix(ckpt.get("model", ckpt))
            print("[load] using student weights")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[load] loading official facebook/VGGT-1B weights from HuggingFace")
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def run_inference(model, image_names: List[str], device: str, dtype) -> dict:
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images
    out = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            out[k] = arr
    return out


# ─────────────────────────────────────────────
# Scene builders
# ─────────────────────────────────────────────

def collect_image_folder_scenes(input_root: str, prefix: str):
    """Yield (scene_name, image_folder) for each sub-directory in input_root."""
    for folder in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, folder)
        if os.path.isdir(full):
            yield prefix + folder, full


def collect_video_scenes(video_dir: str, prefix: str, target_fps: float):
    """Yield (scene_name, frames_dir, video_path) for each video in video_dir."""
    VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
    for fname in sorted(os.listdir(video_dir)):
        if os.path.splitext(fname)[1].lower() in VIDEO_EXTS:
            stem = os.path.splitext(fname)[0]
            video_path = os.path.join(video_dir, fname)
            frames_dir = os.path.join(video_dir, f"{stem}_frames")
            yield prefix + stem, frames_dir, video_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Input: image folders
    parser.add_argument("--input_root", type=str, default=None,
                        help="Root dir with one sub-folder per scene (image folder mode)")
    # Input: videos
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Dir containing video files (mp4/mov) to process")
    parser.add_argument("--target_fps", type=float, default=1.0,
                        help="Frame rate for video extraction (default: 1.0)")
    # Output
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root where <prefix><scene>/cache.npz will be written")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix added to scene name (e.g. 'davis-', 'wild-')")
    # Model
    parser.add_argument("--ckpt_path", type=str, default=None)
    # Sky
    parser.add_argument("--mask_sky", action="store_true",
                        help="Apply sky segmentation (zeros out sky confidence)")
    parser.add_argument("--skyseg_onnx", type=str, default="skyseg.onnx",
                        help="Path to skyseg.onnx model file")
    # Misc
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    if args.input_root is None and args.video_dir is None:
        parser.error("Provide --input_root or --video_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
    print(f"[main] device={device}  dtype={dtype}  mask_sky={args.mask_sky}")

    model = load_model(args.ckpt_path, device)

    # Build scene list
    scenes = []  # list of (name, frames_dir, video_path_or_None)
    if args.input_root:
        for name, folder in collect_image_folder_scenes(args.input_root, args.prefix):
            scenes.append((name, folder, None))
    if args.video_dir:
        for name, frames_dir, video_path in collect_video_scenes(
                args.video_dir, args.prefix, args.target_fps):
            scenes.append((name, frames_dir, video_path))

    print(f"[main] {len(scenes)} scenes to process")

    for idx, (scene_name, frames_dir, video_path) in enumerate(scenes):
        out_path = os.path.join(args.output_root, scene_name, "cache.npz")
        print(f"\n[{idx+1}/{len(scenes)}] {scene_name}")

        if args.skip_existing and os.path.exists(out_path):
            print("  SKIP (cache exists)")
            continue

        # Extract frames from video if needed
        if video_path is not None:
            image_names = extract_frames_from_video(
                video_path, frames_dir, target_fps=args.target_fps)
        else:
            exts = ("*.jpg", "*.jpeg", "*.png")
            image_names = []
            for ext in exts:
                image_names += glob.glob(os.path.join(frames_dir, ext))
            image_names = sort_by_index(image_names)

        if not image_names:
            print("  SKIP (no images found)")
            continue

        print(f"  {len(image_names)} frames")
        try:
            pred = run_inference(model, image_names, device, dtype)

            # Sky masking: zero out confidence for sky pixels
            if args.mask_sky:
                for conf_key in ("world_points_conf", "depth_conf"):
                    if conf_key in pred:
                        pred[conf_key] = apply_sky_segmentation(
                            pred[conf_key], frames_dir, args.skyseg_onnx)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, **pred)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  saved → {out_path}  ({size_mb:.1f} MB)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    print("\n[main] batch inference complete.")


if __name__ == "__main__":
    main()
