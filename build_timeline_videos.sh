#!/bin/bash
#SBATCH -J selfevo_tl_videos
#SBATCH -p all
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_tl_videos.out
#SBATCH -e /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_tl_videos.err

# Render timeline evolution videos (CPU only, no GPU needed).
# Estimated time: ~10-20 minutes total.

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate vggt

SITE=/home/nan.huang/code/page/Self-Evo.github.io
cd $SITE

mkdir -p static/videos/timeline logs

echo "=== Rendering all timeline videos ==="
python tools/render_timeline.py \
    --all \
    --exports_dir exports \
    --output_dir static/videos/timeline \
    --width 1280 --height 720 \
    --max_points 300000 \
    --stride 2 \
    --fps 30 --hold_sec 1.5 --fade_sec 0.5 \
    --point_radius 1

echo "=== Done ==="
ls -lh static/videos/timeline/*.mp4
