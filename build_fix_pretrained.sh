#!/bin/bash
#SBATCH -J selfevo_fix_pretrained
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH -o website/selfevo/logs/%j.out
#SBATCH -e website/selfevo/logs/%j.err

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

REPO=/home/nan.huang/code/vggt
cd $REPO
export PYTHONPATH=$REPO

PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS="website/selfevo/exports"
STATIC="website/selfevo/static"

# Re-pack pretrained dragon scenes with correct frame ranges
echo "=== Re-pack pretrained: dragon-fly2 (frames 0-7) ==="
$PACK --npz_path $EXPORTS/pretrained/dragon-fly2/cache.npz \
      --output $STATIC/packed/pretrained/dragon-fly2.packed \
      --width 512 --frame_start 0 --frame_end 7 \
      --thumbnail --thumb_dir $STATIC/images/thumbs

echo "=== Re-pack pretrained: dragon-battle2 (frames 0-9) ==="
$PACK --npz_path $EXPORTS/pretrained/dragon-battle2/cache.npz \
      --output $STATIC/packed/pretrained/dragon-battle2.packed \
      --width 512 --frame_start 0 --frame_end 9 \
      --thumbnail --thumb_dir $STATIC/images/thumbs

echo "=== Done ==="
