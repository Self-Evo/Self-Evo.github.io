#!/bin/bash
#SBATCH -J selfevo_update
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o website/selfevo/logs/%j.out
#SBATCH -e website/selfevo/logs/%j.err

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

REPO=/home/nan.huang/code/vggt
cd $REPO
export PYTHONPATH=$REPO

INFER="python website/selfevo/tools/run_inference.py"
PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS="website/selfevo/exports"
STATIC="website/selfevo/static"
CKPT_Z=training/logs/zhenhuan_random_995_frcam/ckpts/checkpoint_11.pt
CKPT_D=training/logs/dragon_random_995_frcam/ckpts/checkpoint_3.pt

# ============================================================
# zhenhuan-3: re-run inference (frames were deleted from folder)
# ============================================================
echo "=== Re-inference: zhenhuan-3 ==="
$INFER --image_folder examples/movie/zhenhuan/3 \
       --output $EXPORTS/pretrained/zhenhuan-3/cache.npz

$INFER --image_folder examples/movie/zhenhuan/3 --ckpt_path $CKPT_Z \
       --output $EXPORTS/selfevo/zhenhuan-3/cache.npz

echo "=== Re-pack: zhenhuan-3 ==="
for VERSION in pretrained selfevo; do
  $PACK --npz_path $EXPORTS/$VERSION/zhenhuan-3/cache.npz \
        --output $STATIC/packed/$VERSION/zhenhuan-3.packed --width 512
done

# ============================================================
# dragon-fly2: re-pack from existing cache, frames 0-7
# ============================================================
echo "=== Re-pack: dragon-fly2 (frames 0-7) ==="
for VERSION in pretrained selfevo; do
  $PACK --npz_path $EXPORTS/$VERSION/dragon-fly2/cache.npz \
        --output $STATIC/packed/$VERSION/dragon-fly2.packed \
        --width 512 --frame_start 0 --frame_end 7
done

# ============================================================
# dragon-battle2: re-pack from existing cache, frames 0-9
# ============================================================
echo "=== Re-pack: dragon-battle2 (frames 0-9) ==="
for VERSION in pretrained selfevo; do
  $PACK --npz_path $EXPORTS/$VERSION/dragon-battle2/cache.npz \
        --output $STATIC/packed/$VERSION/dragon-battle2.packed \
        --width 512 --frame_start 0 --frame_end 9
done

# ============================================================
# Regenerate thumbnails for updated scenes
# (pass same frame ranges to avoid overwriting trimmed packed files)
# ============================================================
echo "=== Thumbnails ==="
$PACK --npz_path $EXPORTS/pretrained/zhenhuan-3/cache.npz \
      --output $STATIC/packed/pretrained/zhenhuan-3.packed \
      --width 512 --thumbnail --thumb_dir $STATIC/images/thumbs

$PACK --npz_path $EXPORTS/pretrained/dragon-fly2/cache.npz \
      --output $STATIC/packed/pretrained/dragon-fly2.packed \
      --width 512 --frame_start 0 --frame_end 7 \
      --thumbnail --thumb_dir $STATIC/images/thumbs

$PACK --npz_path $EXPORTS/pretrained/dragon-battle2/cache.npz \
      --output $STATIC/packed/pretrained/dragon-battle2.packed \
      --width 512 --frame_start 0 --frame_end 9 \
      --thumbnail --thumb_dir $STATIC/images/thumbs

echo "=== Done ==="
