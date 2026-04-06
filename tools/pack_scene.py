#!/usr/bin/env python3
"""
Pack VGGT predictions into the .packed binary format for the MegaSaM-style WebGL viewer.

Usage:
    # Pack from a prediction cache (.npz)
    python pack_scene.py --npz_path cache.npz --output scene.packed --width 512

    # Pack all scenes from scene_list.json
    python pack_scene.py --config scene_list.json --width 512 --output_dir static/packed

    # Run inference + pack (requires GPU)
    python pack_scene.py --video_path video.mp4 --ckpt_path ckpt.pt --output scene.packed --width 512
"""

import argparse
import io
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def closed_form_inverse_se3(extrinsics: np.ndarray) -> np.ndarray:
    """Invert SE3 matrices: world-to-camera (S,3,4) -> camera-to-world (S,4,4)."""
    R = extrinsics[..., :3, :3]  # (S, 3, 3)
    t = extrinsics[..., :3, 3:]  # (S, 3, 1)
    R_inv = np.swapaxes(R, -2, -1)  # transpose
    t_inv = -R_inv @ t
    bottom = np.zeros_like(extrinsics[..., :1, :])  # (S, 1, 4)
    bottom[..., 0, 3] = 1.0
    top = np.concatenate([R_inv, t_inv], axis=-1)  # (S, 3, 4)
    return np.concatenate([top, bottom], axis=-2)  # (S, 4, 4)


def encode_depth_png(depth: np.ndarray, depth_scale: float = 20.0) -> bytes:
    """
    Encode a depth map into RGBA PNG using the MegaSaM convention.

    In the WebGL shader:
        depth = depth_scale * (rgba.r + rgba.g / 256.0)
    where rgba values are 0-1 (uint8 / 255).

    Args:
        depth: (H, W) depth map in world units
        depth_scale: normalization factor (default 20)

    Returns:
        PNG bytes
    """
    H, W = depth.shape
    val = np.clip(depth / depth_scale, 0.0, 1.0)  # normalize to 0-1

    val_scaled = val * 255.0
    r_byte = np.floor(val_scaled).astype(np.uint8)
    g_byte = np.clip(np.round((val_scaled - r_byte.astype(np.float64)) * 256.0), 0, 255).astype(np.uint8)

    # RGBA image: depth in R+G, zeros in B, 255 in A
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, 0] = r_byte
    rgba[:, :, 1] = g_byte
    rgba[:, :, 3] = 255

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB image (H, W, 3) uint8 to PNG bytes."""
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def resize_with_aspect(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image keeping aspect ratio, with target width."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(round(h * scale))
    # Ensure even dimensions
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    target_width = target_width if target_width % 2 == 0 else target_width + 1
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def build_packed(data_json: dict, files: dict) -> bytes:
    """
    Build the .packed binary format.

    Format:
        [0-4]   uint32 prefix_size (little-endian)
        [4-8]   padding zeros
        [8..prefix_size] JSON index (UTF-8)
        [prefix_size..] concatenated file blobs

    The JSON index maps filename -> [start_offset, end_offset, content_type]
    where offsets are relative to the end of the JSON block.
    """
    # Serialize data.json first
    data_json_bytes = json.dumps(data_json, separators=(",", ":")).encode("utf-8")

    # Build all blobs: data.json + rgb PNGs + depth PNGs
    blobs = [("data.json", data_json_bytes, "application/json")]
    for name, content, ctype in files:
        blobs.append((name, content, ctype))

    # Compute offsets
    index = {}
    offset = 0
    for name, content, ctype in blobs:
        end = offset + len(content)
        index[name] = [offset, end, ctype]
        offset = end

    # Serialize index JSON
    index_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")

    # prefix_size = 8 (header) + len(index_bytes)
    prefix_size = 8 + len(index_bytes)

    # Build binary
    header = struct.pack("<I", prefix_size) + b"\x00" * 4
    blob_data = b"".join(content for _, content, _ in blobs)

    return header + index_bytes + blob_data


def pack_from_npz(npz_path: str, output_path: str, target_width: int = 512,
                  depth_scale: float = 20.0, frame_start: int = 0, frame_end: int = -1,
                  apply_conf_mask: bool = True):
    """
    Pack a VGGT prediction cache (.npz) into .packed format.
    """
    print(f"Loading predictions from {npz_path}")
    cache = np.load(npz_path, allow_pickle=True)

    images = cache["images"]          # (S, 3, H, W)
    depth = cache["depth"]            # (S, H, W, 1) or (S, H, W)
    extrinsics = cache["extrinsic"]   # (S, 3, 4)
    intrinsics = cache["intrinsic"]   # (S, 3, 3)

    S = images.shape[0]

    # Apply frame range
    if frame_end == -1:
        frame_end = S
    frame_end = min(frame_end, S)
    frame_start = max(0, frame_start)

    images = images[frame_start:frame_end]
    depth = depth[frame_start:frame_end]
    extrinsics = extrinsics[frame_start:frame_end]
    intrinsics = intrinsics[frame_start:frame_end]
    S = images.shape[0]

    print(f"  Frames: {S}, Original shape: {images.shape}")

    # depth: ensure (S, H, W)
    if depth.ndim == 4:
        depth = depth.squeeze(-1)

    # Apply confidence mask: zero out sky/low-confidence pixels so the WebGL
    # shader discards them (z=0 → gl_PointSize=0 → invisible point).
    if apply_conf_mask:
        for conf_key in ("depth_conf", "world_points_conf"):
            if conf_key in cache:
                conf = cache[conf_key][frame_start:frame_end]
                if conf.ndim == 4:
                    conf = conf.squeeze(-1)
                depth = np.where(conf > 0, depth, 0.0)
                break

    # images: (S, 3, H, W) -> (S, H, W, 3), convert to uint8
    imgs_hwc = np.transpose(images, (0, 2, 3, 1))  # (S, H, W, 3)
    if imgs_hwc.max() <= 1.0:
        imgs_hwc = (imgs_hwc * 255).clip(0, 255).astype(np.uint8)
    else:
        imgs_hwc = imgs_hwc.clip(0, 255).astype(np.uint8)

    _, H_orig, W_orig, _ = imgs_hwc.shape

    # Convert extrinsics to camera-to-world
    cam_to_world = closed_form_inverse_se3(extrinsics)  # (S, 4, 4)

    # Center the scene
    # Use median of camera positions as scene center for stability
    cam_positions = cam_to_world[:, :3, 3]  # (S, 3)
    scene_center = np.median(cam_positions, axis=0)
    cam_to_world[:, :3, 3] -= scene_center[None, :]

    # Resize images and depth
    scale = target_width / W_orig
    new_H = int(round(H_orig * scale))
    new_H = new_H if new_H % 2 == 0 else new_H + 1
    new_W = target_width if target_width % 2 == 0 else target_width + 1

    # Adjust depth_scale based on scene scale
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    if valid_depth.size > 0:
        p99 = np.percentile(valid_depth, 99)
        # depth_scale should be large enough that max_depth / depth_scale <= 1
        depth_scale = max(depth_scale, float(np.ceil(p99 * 1.1)))
    print(f"  Using depth_scale={depth_scale:.1f}")

    # Convert intrinsics from pixel 3x3 to normalized [fx, fy, cx, cy]
    # VGGT intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] in pixel coords
    # MegaSaM expects [fx, fy, cx, cy] normalized to UV space (0-1)
    poses_list = []
    intrinsics_list = []

    files = []
    for i in range(S):
        # Resize RGB
        rgb_resized = resize_with_aspect(imgs_hwc[i], target_width)
        actual_H, actual_W = rgb_resized.shape[:2]

        # Resize depth
        depth_i = depth[i]
        depth_resized = cv2.resize(depth_i.astype(np.float32), (actual_W, actual_H),
                                   interpolation=cv2.INTER_NEAREST)

        # Encode RGB PNG
        rgb_bytes = encode_rgb_png(rgb_resized)
        files.append((f"rgb_{i+1:05d}.png", rgb_bytes, "image/png"))

        # Encode depth PNG
        depth_bytes = encode_depth_png(depth_resized, depth_scale)
        files.append((f"depthrgb_{i+1:05d}.png", depth_bytes, "image/png"))

        # Convert pose (camera-to-world, 4x4) -> 3x4 list
        pose_3x4 = cam_to_world[i, :3, :].tolist()
        poses_list.append(pose_3x4)

        # Convert intrinsic
        K = intrinsics[i]  # 3x3 pixel coords
        fx_px, fy_px = K[0, 0], K[1, 1]
        cx_px, cy_px = K[0, 2], K[1, 2]

        # Normalize to UV (0-1) based on original image dimensions
        fx_norm = fx_px / W_orig
        fy_norm = fy_px / H_orig
        cx_norm = cx_px / W_orig
        cy_norm = cy_px / H_orig
        intrinsics_list.append([float(fx_norm), float(fy_norm), float(cx_norm), float(cy_norm)])

    data_json = {
        "poses": poses_list,
        "intrinsics": intrinsics_list,
        "width": actual_W,
        "height": actual_H,
    }

    packed_bytes = build_packed(data_json, files)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(packed_bytes)

    size_mb = len(packed_bytes) / (1024 * 1024)
    print(f"  Written {output_path} ({size_mb:.1f} MB, {S} frames)")
    return output_path


def pack_from_config(config_path: str, output_dir: str, target_width: int = 512,
                     depth_scale: float = 20.0, repo_root: str = None):
    """
    Pack all scenes defined in scene_list.json.
    Expects prediction caches to already exist at exports/<version>/<scene_name>/cache.npz.
    """
    with open(config_path) as f:
        config = json.load(f)

    if repo_root is None:
        repo_root = str(Path(config_path).parent)

    exports_dir = os.path.join(repo_root, "exports")

    for section_name in ["adaptation", "generalization"]:
        section = config.get(section_name, {})
        scenes = section.get("scenes", [])
        shared_ckpt = section.get("ckpt_path", None)

        for scene in scenes:
            name = scene["name"]
            frame_start = scene.get("frame_start_idx", 0)
            frame_end = scene.get("frame_end_idx", -1)

            # Try to find existing cache
            for version in ["pretrained", "selfevo"]:
                npz_path = os.path.join(exports_dir, version, name, "cache.npz")
                out_path = os.path.join(output_dir, version, f"{name}.packed")

                if os.path.exists(npz_path):
                    print(f"\n[{section_name}] Packing {version}/{name}")
                    pack_from_npz(npz_path, out_path, target_width, depth_scale,
                                  frame_start, frame_end)
                else:
                    print(f"[{section_name}] SKIP {version}/{name} — cache not found at {npz_path}")


def generate_thumbnail(npz_path: str, output_path: str, target_width: int = 200, target_height: int = 140, frame_index: int = 0):
    """Extract a frame as a JPEG thumbnail. Use frame_index=-1 for middle frame."""
    cache = np.load(npz_path, allow_pickle=True)
    images = cache["images"]  # (S, 3, H, W)
    if frame_index == -1:
        frame_index = len(images) // 2
    img = np.transpose(images[frame_index], (1, 2, 0))  # (H, W, 3)
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)

    # Resize and crop to target
    h, w = img.shape[:2]
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    y0 = (new_h - target_height) // 2
    x0 = (new_w - target_width) // 2
    img = img[y0:y0+target_height, x0:x0+target_width]

    img_pil = Image.fromarray(img)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img_pil.save(output_path, "JPEG", quality=80)
    print(f"  Thumbnail: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pack VGGT predictions into .packed format")
    parser.add_argument("--npz_path", type=str, help="Path to prediction cache (.npz)")
    parser.add_argument("--output", type=str, help="Output .packed file path")
    parser.add_argument("--config", type=str, help="Path to scene_list.json for batch packing")
    parser.add_argument("--output_dir", type=str, default="static/packed",
                        help="Output directory for batch packing")
    parser.add_argument("--width", type=int, default=512, help="Target image width")
    parser.add_argument("--depth_scale", type=float, default=20.0, help="Depth scale factor")
    parser.add_argument("--frame_start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frame_end", type=int, default=-1, help="End frame index (-1=all)")
    parser.add_argument("--thumbnail", action="store_true", help="Also generate thumbnail")
    parser.add_argument("--thumb_dir", type=str, default="static/images/thumbs",
                        help="Thumbnail output directory")
    parser.add_argument("--no_conf_mask", action="store_true",
                        help="Skip confidence masking (show all pixels including sky)")

    args = parser.parse_args()

    if args.config:
        pack_from_config(args.config, args.output_dir, args.width, args.depth_scale)
    elif args.npz_path and args.output:
        pack_from_npz(args.npz_path, args.output, args.width, args.depth_scale,
                      args.frame_start, args.frame_end,
                      apply_conf_mask=not args.no_conf_mask)
        if args.thumbnail:
            name = Path(args.output).stem
            thumb_path = os.path.join(args.thumb_dir, f"{name}.jpg")
            generate_thumbnail(args.npz_path, thumb_path)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Pack a single scene")
        print("  python pack_scene.py --npz_path exports/pretrained/zhenhuan-dance2/cache.npz \\")
        print("    --output static/packed/pretrained/zhenhuan-dance2.packed --width 512")
        print()
        print("  # Pack all scenes from config")
        print("  python pack_scene.py --config ../../scene_list.json --output_dir ../static/packed")
        sys.exit(1)


if __name__ == "__main__":
    main()
