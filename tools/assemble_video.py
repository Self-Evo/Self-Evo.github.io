#!/usr/bin/env python3
"""Assemble PNG frames from a zip file into an MP4 video.

Usage:
    python tools/assemble_video.py split_recording.zip -o tweet_videos/split_comparison.mp4 --fps 30
"""
import argparse
import zipfile
import io
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Assemble frames zip into MP4")
    parser.add_argument("zip_path", help="Path to zip file containing frame_XXXXX.png files")
    parser.add_argument("-o", "--output", default="split_comparison.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        names = sorted([n for n in zf.namelist() if n.endswith(".png")])
        print(f"Found {len(names)} frames")

        writer = None
        for i, name in enumerate(names):
            data = zf.read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )
                print(f"Resolution: {w}x{h}, FPS: {args.fps}")
            writer.write(img)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} / {len(names)}")

        if writer:
            writer.release()
            print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
