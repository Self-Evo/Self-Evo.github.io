#!/usr/bin/env python3
"""
Render timeline evolution videos from cached NPZ predictions.

For each scene, renders a static point-cloud + camera-frusta image per checkpoint,
then stitches them into a crossfade video.

Usage:
    # Render one scene
    python tools/render_timeline.py --scene wild-tom1

    # Render all scenes
    python tools/render_timeline.py --all

    # Preview a single checkpoint (no video, just show image)
    python tools/render_timeline.py --scene wild-tom1 --preview --ckpt pretrain
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────
# Checkpoint versions
# ──────────────────────────────────────────────────��──────────
VERSIONS = [
    ("timeline_pretrain", "Pretrained"),
    ("timeline_ckpt_00", "Iter 50"),
    ("timeline_ckpt_01", "Iter 100"),
    ("timeline_ckpt_02", "Iter 150"),
    ("timeline_ckpt_03", "Iter 200"),
    ("timeline_ckpt_04", "Iter 250"),
    ("timeline_ckpt_05", "Iter 300"),
    ("timeline_ckpt_06", "Iter 350"),
    ("timeline_ckpt_07", "Iter 400"),
    ("timeline_ckpt_08", "Iter 450"),
    ("timeline_ckpt_09", "Iter 500"),
    ("timeline_ckpt_10", "Iter 550"),
    ("timeline_ckpt_11", "Iter 600"),
    ("timeline_ckpt_12", "Iter 650"),
]

# ─────────────────────────────────────────────────────────────
# Per-scene camera presets
# Fields: distance, rx, ry, tx, ty, elevation (matching viewer.js defaults)
# ─────────────────────────────────────────────────────────────
SCENE_CAMERAS = {
    "wild-tom1": dict(distance=0.367, rx=0.0, ry=0.0, tx=-0.3681, ty=-0.1805, elevation=0.1),
    "wild-5207050-uhd_3840_2160_25fps": dict(distance=3.51, rx=0.4461, ry=0.0088, tx=0.0, ty=1.053, elevation=0.1),
    "wild-6539141-hd_1920_1080_25fps2": dict(distance=0.535, rx=0.3296, ry=0.4632, tx=0.0, ty=-0.2408, elevation=0.1),
    "wild-penguin": dict(distance=0.4, rx=0.0, ry=0.0, tx=0.0, ty=0.0, elevation=0.1),
    "davis-kid-football": dict(distance=0.275, rx=0.4699, ry=0.01652, tx=0.0, ty=-0.18825, elevation=0.1),
    "wild2-corgi-snow-2": dict(distance=0.4, rx=0.3291, ry=-0.0085, tx=0.0, ty=-0.18, elevation=0.1),
}

# Frustum rainbow color stops (tech gradient from viewer.js)
FRUSTUM_COLORS = [
    [0.529, 0.176, 0.918],
    [0.400, 0.200, 0.937],
    [0.282, 0.282, 0.945],
    [0.200, 0.400, 0.933],
    [0.149, 0.537, 0.898],
    [0.118, 0.671, 0.839],
    [0.098, 0.788, 0.765],
    [0.118, 0.878, 0.698],
]


def rainbow_color(t):
    """Interpolate frustum color from the tech gradient stops."""
    stops = FRUSTUM_COLORS
    n = len(stops) - 1
    s = min(t * n, n - 1e-6)
    i = int(s)
    f = s - i
    a, b = stops[i], stops[i + 1]
    return [a[j] + (b[j] - a[j]) * f for j in range(3)]


# ─────────────────────────────────────────────────────────────
# 4x4 matrix helpers (row-major, matching viewer.js conventions)
# ─────────────────────────────────────────────────────────────
def mat_identity():
    return np.eye(4, dtype=np.float64)


def mat_translate(x, y, z):
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
    return m


def mat_rx(t):
    c, s = np.cos(t), np.sin(t)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c
    return m


def mat_ry(t):
    c, s = np.cos(t), np.sin(t)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m


def mat_scale(t):
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = t; m[1, 1] = t; m[2, 2] = t
    return m


def closed_form_inverse_se3(extrinsics):
    """Invert SE3: world-to-camera (S,3,4) -> camera-to-world (S,4,4)."""
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3:]
    R_inv = np.swapaxes(R, -2, -1)
    t_inv = -R_inv @ t
    bottom = np.zeros_like(extrinsics[..., :1, :])
    bottom[..., 0, 3] = 1.0
    top = np.concatenate([R_inv, t_inv], axis=-1)
    return np.concatenate([top, bottom], axis=-2)


# ─────────────────────────────────────────────────────────────
# Camera matrix (replicates viewer.js cameraMatrix)
# ─────────────────────────────────────────────────────────────
def build_camera_matrix(cam, first_cam_pos, aspect_ratio, near=0.01, far=100.0):
    """
    Build view-projection matrix matching the viewer.js camera model.

    cam: dict with distance, rx, ry, tx, ty, elevation, zoom
    first_cam_pos: (3,) position of the first camera (for follow_position)
    aspect_ratio: height / width
    """
    d = 1.0 / (far - near)
    a = (near + far) * d
    b = -2.0 * near * far * d
    zoom = cam.get("zoom", 1.0)
    w = zoom
    h = zoom * aspect_ratio

    perspective = np.array([
        [w, 0, 0, 0],
        [0, -h, 0, 0],
        [0, 0, a, b],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    dist = cam["distance"]
    elev = cam.get("elevation", 0.1)
    rx = cam.get("rx", 0.0)
    ry = cam.get("ry", 0.0)
    tx = cam.get("tx", 0.0)
    ty = cam.get("ty", 0.0)
    fwd = cam.get("forward", 0.0)

    T_elevation = mat_translate(0, elev * dist, dist)
    Rx = mat_rx(rx)
    Ry = mat_ry(ry)
    T_target = mat_translate(-tx, -ty, -fwd)
    T_follow = mat_translate(-first_cam_pos[0], -first_cam_pos[1], -first_cam_pos[2])

    return perspective @ T_elevation @ Rx @ Ry @ T_target @ T_follow


def project_points(pts, mvp, width, height):
    """
    Project Nx3 world points through 4x4 MVP matrix to screen coords.
    Returns Nx3 (screen_x, screen_y, clip_z) and mask of visible points.
    """
    N = pts.shape[0]
    ones = np.ones((N, 1), dtype=pts.dtype)
    pts4 = np.hstack([pts, ones])  # (N, 4)

    clip = (mvp @ pts4.T).T  # (N, 4)
    w = clip[:, 3]

    # Clip behind camera
    visible = w > 0.001

    ndc = np.zeros((N, 3), dtype=np.float64)
    ndc[visible, 0] = clip[visible, 0] / w[visible]
    ndc[visible, 1] = clip[visible, 1] / w[visible]
    ndc[visible, 2] = clip[visible, 2] / w[visible]

    # NDC to screen: x in [-1,1] -> [0, width], y in [-1,1] -> [0, height]
    screen = np.zeros((N, 3), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] + 1) * 0.5 * width
    screen[:, 1] = (ndc[:, 1] + 1) * 0.5 * height
    screen[:, 2] = w  # use clip w for depth sorting

    return screen, visible


# ───────────────────────────────────────────────────────────���─
# Point cloud rendering
# ─────────────────────────────────────────────────────────────
def render_pointcloud(world_pts, colors, mvp, width, height, bg_color=(242, 242, 242),
                      point_radius=1):
    """
    Render a point cloud to an image using vectorized numpy splatting.

    world_pts: (N, 3)
    colors: (N, 3) uint8
    Returns: (height, width, 3) uint8 image
    """
    screen, visible = project_points(world_pts, mvp, width, height)

    # Filter to visible and in-bounds
    sx = screen[visible, 0]
    sy = screen[visible, 1]
    sz = screen[visible, 2]
    cols = colors[visible]

    in_bounds = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height) & (sz > 0)
    sx = sx[in_bounds]
    sy = sy[in_bounds]
    sz = sz[in_bounds]
    cols = cols[in_bounds]

    # Sort far-to-near (so near points overwrite far ones)
    order = np.argsort(-sz)
    px = sx[order].astype(np.int32)
    py = sy[order].astype(np.int32)
    cols_sorted = cols[order]

    # Create image with background
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Splat points (vectorized: last write wins = nearest point)
    if point_radius <= 1:
        img[py, px] = cols_sorted
    else:
        # Expand each point to a small square for visibility
        for dr in range(-point_radius, point_radius + 1):
            for dc in range(-point_radius, point_radius + 1):
                if dr * dr + dc * dc > point_radius * point_radius:
                    continue
                qy = np.clip(py + dr, 0, height - 1)
                qx = np.clip(px + dc, 0, width - 1)
                img[qy, qx] = cols_sorted

    return img


# ─────────────────────────────────────────────────────────────
# Frustum drawing
# ─────────────────────────────────────────────────────────────
FRUSTUM_LINES = [
    # origin to 4 corners
    (0, 1), (0, 2), (0, 3), (0, 4),
    # rectangle connecting corners
    (1, 2), (2, 4), (4, 3), (3, 1),
]


def get_frustum_world_points(c2w, intrinsic, frustum_size=0.15):
    """
    Get 5 frustum vertices in world space.
    Vertices: 0=origin, 1-4=corners at frustum_size depth.
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Image corners in normalized coords at unit depth
    # The viewer uses (0,0,1), (1,0,1), (0,1,1), (1,1,1) in a unit cube
    # scaled by frustum_size, then transformed by pose
    # We replicate this: corners at (0,0,f), (f,0,f), (0,f,f), (f,f,f) then pose
    f = frustum_size
    local_pts = np.array([
        [0, 0, 0],      # origin
        [0, 0, f],       # corner 0,0
        [f, 0, f],       # corner 1,0
        [0, f, f],       # corner 0,1
        [f, f, f],       # corner 1,1
    ], dtype=np.float64)

    # Transform to world space
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    world_pts = (R @ local_pts.T).T + t
    return world_pts


def draw_frusta_on_image(img, c2w_all, intrinsics, mvp, width, height,
                         frustum_size=0.15, line_thickness=1):
    """Draw camera frusta wireframes on the image."""
    S = c2w_all.shape[0]
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    for i in range(S):
        t = i / max(S - 1, 1)
        color_f = rainbow_color(t)
        color = tuple(int(c * 255) for c in color_f)

        pts_world = get_frustum_world_points(c2w_all[i], intrinsics[i], frustum_size)
        screen, visible = project_points(pts_world, mvp, width, height)

        for a, b in FRUSTUM_LINES:
            if visible[a] and visible[b]:
                x0, y0 = screen[a, 0], screen[a, 1]
                x1, y1 = screen[b, 0], screen[b, 1]
                # Check bounds loosely
                if (max(x0, x1) >= -width and min(x0, x1) < 2 * width and
                        max(y0, y1) >= -height and min(y0, y1) < 2 * height):
                    draw.line([(x0, y0), (x1, y1)], fill=color, width=line_thickness)

    return np.array(img_pil)


# ─────────────────────────────────────────────────────────────
# Label overlay
# ─────────────────────────────────────────────────────────────
def overlay_label(img, text, position="top-left", font_size=24):
    """Burn iteration label onto the image."""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad = 8
    if position == "top-left":
        x, y = 16, 16
    elif position == "top-center":
        x, y = (img.shape[1] - tw) // 2, 16
    else:
        x, y = 16, 16

    # Semi-transparent background
    draw.rounded_rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        radius=6, fill=(0, 0, 0, 140)
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img_pil.convert("RGB"))


# ─────────────────────────────────────────────────────────────
# Scene rendering
# ─────────────────────────────────────────────────────────────
def load_scene(npz_path, max_points=300000, stride=2):
    """Load scene data and subsample points."""
    cache = np.load(npz_path, allow_pickle=True)

    images = cache["images"]          # (S, 3, H, W)
    world_pts = cache["world_points"]  # (S, H, W, 3)
    extrinsic = cache["extrinsic"]    # (S, 3, 4)
    intrinsic = cache["intrinsic"]    # (S, 3, 3)

    S, _, H, W = images.shape

    # Get confidence mask
    conf = None
    for key in ("world_points_conf", "depth_conf"):
        if key in cache:
            conf = cache[key]
            if conf.ndim == 4:
                conf = conf.squeeze(-1)
            break

    # Subsample spatially
    pts_sub = world_pts[:, ::stride, ::stride, :]     # (S, H', W', 3)
    img_sub = images[:, :, ::stride, ::stride]        # (S, 3, H', W')
    conf_sub = conf[:, ::stride, ::stride] if conf is not None else None

    # Flatten to (N, 3) and (N, 3)
    S2, H2, W2, _ = pts_sub.shape
    pts_flat = pts_sub.reshape(-1, 3)
    colors_flat = np.transpose(img_sub, (0, 2, 3, 1)).reshape(-1, 3)  # (N, 3) float

    # Convert colors to uint8
    if colors_flat.max() <= 1.0:
        colors_flat = (colors_flat * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_flat = colors_flat.clip(0, 255).astype(np.uint8)

    # Apply confidence mask
    if conf_sub is not None:
        valid = conf_sub.reshape(-1) > 0
        pts_flat = pts_flat[valid]
        colors_flat = colors_flat[valid]

    # Filter invalid points
    finite = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[finite]
    colors_flat = colors_flat[finite]

    # Random subsample if still too many
    if pts_flat.shape[0] > max_points:
        idx = np.random.choice(pts_flat.shape[0], max_points, replace=False)
        pts_flat = pts_flat[idx]
        colors_flat = colors_flat[idx]

    # Camera-to-world
    c2w = closed_form_inverse_se3(extrinsic)  # (S, 4, 4)

    # Center scene at median camera position
    cam_positions = c2w[:, :3, 3]
    scene_center = np.median(cam_positions, axis=0)
    pts_flat -= scene_center
    c2w[:, :3, 3] -= scene_center

    return pts_flat, colors_flat, c2w, intrinsic, cam_positions[0] - scene_center


def render_checkpoint(npz_path, cam_config, width, height, max_points, stride,
                      label=None, frustum_size=0.15, point_radius=1):
    """Render a single checkpoint as an image."""
    pts, colors, c2w, intrinsics, first_cam = load_scene(npz_path, max_points, stride)

    aspect = height / width
    mvp = build_camera_matrix(cam_config, first_cam, aspect)

    img = render_pointcloud(pts, colors, mvp, width, height, point_radius=point_radius)
    img = draw_frusta_on_image(img, c2w, intrinsics, mvp, width, height,
                                frustum_size=frustum_size, line_thickness=2)

    if label:
        img = overlay_label(img, label, position="top-left", font_size=max(20, height // 30))

    return img


# ─────────────────────────────────────────────────────────────
# Video stitching with crossfade
# ─────────────────────────────────────────────────────────────
def stitch_video(frames, output_path, fps=30, hold_sec=1.5, fade_sec=0.5):
    """
    Create a crossfade video from a list of images.

    frames: list of (H, W, 3) uint8 numpy arrays
    """
    hold_frames = int(hold_sec * fps)
    fade_frames = int(fade_sec * fps)

    tmpdir = tempfile.mkdtemp()
    frame_idx = 0

    for i, img in enumerate(frames):
        # Hold frames
        for _ in range(hold_frames):
            cv2.imwrite(os.path.join(tmpdir, f"{frame_idx:05d}.png"), img[:, :, ::-1])
            frame_idx += 1

        # Crossfade to next (except for last frame)
        if i < len(frames) - 1:
            next_img = frames[i + 1]
            for f in range(fade_frames):
                alpha = (f + 1) / (fade_frames + 1)
                blended = cv2.addWeighted(img, 1 - alpha, next_img, alpha, 0)
                cv2.imwrite(os.path.join(tmpdir, f"{frame_idx:05d}.png"), blended[:, :, ::-1])
                frame_idx += 1

    print(f"  Generated {frame_idx} frames")

    # Encode with ffmpeg
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-r", str(fps),
        "-i", os.path.join(tmpdir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "slow",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Video: {output_path}")

    # Cleanup
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def render_scene(scene_name, exports_dir, output_dir, width=1280, height=720,
                 max_points=300000, stride=2, fps=30, hold_sec=1.5, fade_sec=0.5,
                 point_radius=1, frustum_size=0.15):
    """Render all checkpoints for a scene and stitch into a video."""
    cam = SCENE_CAMERAS.get(scene_name)
    if cam is None:
        print(f"  WARNING: No camera preset for '{scene_name}', using defaults")
        cam = dict(distance=0.4, rx=0.0, ry=0.0, tx=0.0, ty=0.0, elevation=0.1)

    frames = []
    for version_dir, label in VERSIONS:
        npz_path = os.path.join(exports_dir, version_dir, scene_name, "cache.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {version_dir}/{scene_name} — not found")
            continue
        print(f"  Rendering {label} ...")
        img = render_checkpoint(npz_path, cam, width, height, max_points, stride,
                                label=label, frustum_size=frustum_size,
                                point_radius=point_radius)
        frames.append(img)

    if not frames:
        print(f"  ERROR: No checkpoints found for {scene_name}")
        return

    output_path = os.path.join(output_dir, f"{scene_name}.mp4")
    print(f"  Stitching {len(frames)} checkpoints into video ...")
    stitch_video(frames, output_path, fps=fps, hold_sec=hold_sec, fade_sec=fade_sec)


def main():
    parser = argparse.ArgumentParser(description="Render timeline evolution videos")
    parser.add_argument("--scene", type=str, help="Scene name (e.g., wild-tom1)")
    parser.add_argument("--all", action="store_true", help="Render all scenes")
    parser.add_argument("--exports_dir", type=str, default="exports",
                        help="Root exports directory")
    parser.add_argument("--output_dir", type=str, default="static/videos/timeline",
                        help="Output directory for videos")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold_sec", type=float, default=1.5)
    parser.add_argument("--fade_sec", type=float, default=0.5)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--frustum_size", type=float, default=0.15)
    parser.add_argument("--preview", action="store_true",
                        help="Preview single checkpoint (no video)")
    parser.add_argument("--ckpt", type=str, default="pretrain",
                        help="Checkpoint to preview (with --preview)")
    args = parser.parse_args()

    if args.preview:
        scene = args.scene or "wild-tom1"
        ckpt_key = f"timeline_{args.ckpt}"
        label = next((l for k, l in VERSIONS if k == ckpt_key), args.ckpt)
        npz_path = os.path.join(args.exports_dir, ckpt_key, scene, "cache.npz")
        cam = SCENE_CAMERAS.get(scene, dict(distance=0.4, rx=0.0, ry=0.0, tx=0.0, ty=0.0, elevation=0.1))
        print(f"Preview: {scene} @ {label}")
        img = render_checkpoint(npz_path, cam, args.width, args.height,
                                args.max_points, args.stride, label=label,
                                frustum_size=args.frustum_size,
                                point_radius=args.point_radius)
        out = f"preview_{scene}_{args.ckpt}.png"
        Image.fromarray(img).save(out)
        print(f"Saved: {out}")
        return

    scenes = list(SCENE_CAMERAS.keys()) if args.all else [args.scene]
    if not scenes or scenes == [None]:
        parser.error("Specify --scene or --all")

    for scene in scenes:
        print(f"=== {scene} ===")
        render_scene(scene, args.exports_dir, args.output_dir,
                     width=args.width, height=args.height,
                     max_points=args.max_points, stride=args.stride,
                     fps=args.fps, hold_sec=args.hold_sec, fade_sec=args.fade_sec,
                     point_radius=args.point_radius, frustum_size=args.frustum_size)


if __name__ == "__main__":
    main()
