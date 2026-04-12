#!/bin/bash
#SBATCH -J movie4_inference
#SBATCH -p slinky
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/movie4_%j.out
#SBATCH -e logs/movie4_%j.err

set -e

source /home/nan.huang/miniconda3/etc/profile.d/conda.sh
conda activate selfevo

REPO=/home/nan.huang/code/vggt-exp
WEB=/data/home/nan.huang/code/page/Self-Evo.github.io
cd $REPO
export PYTHONPATH=$REPO

INFER="python $WEB/tools/run_inference.py"
BATCH="python $WEB/tools/run_inference_batch.py"
PACK="python $WEB/tools/pack_scene.py"
EXPORTS=$WEB/exports
STATIC=$WEB/static

CKPT=/home/nan.huang/code/vggt-exp/training/logs/random_995_frcam/ckpts/checkpoint_11.pt

# ============================================================
# Phase 1: Pretrained inference (movie4v2 - 3 new clips)
# ============================================================
echo "=== Phase 1: Pretrained movie4v2 ==="
$BATCH \
  --video_dir $WEB/examples/movie4v2 \
  --output_root $EXPORTS/pretrained \
  --prefix movie4v2- \
  --target_fps 10 \
  --mask_sky

# ============================================================
# Phase 2: SelfEvo inference (movie4v2 - 3 new clips)
# ============================================================
echo "=== Phase 2: SelfEvo movie4v2 ==="
$BATCH \
  --video_dir $WEB/examples/movie4v2 \
  --output_root $EXPORTS/selfevo \
  --prefix movie4v2- \
  --ckpt_path $CKPT \
  --target_fps 10 \
  --mask_sky

# ============================================================
# Phase 3: Pack movie4v2 scenes
# ============================================================
echo "=== Phase 3: Packing movie4v2 scenes ==="

MOVIE4_SCENES=(
  movie4v2-1_ori-v2 movie4v2-2_ori-v2 movie4v2-2_ori-v4
)

for SCENE in "${MOVIE4_SCENES[@]}"; do
  for VERSION in pretrained selfevo; do
    NPZ=$EXPORTS/$VERSION/$SCENE/cache.npz
    OUT=$STATIC/packed/$VERSION/$SCENE.packed
    if [ -f "$NPZ" ]; then
      echo "  Packing $VERSION/$SCENE"
      $PACK --npz_path $NPZ --output $OUT --width 512
    else
      echo "  WARNING: $VERSION/$SCENE — cache not found, skipping"
    fi
  done
done

# ============================================================
# Phase 4: Thumbnails (from pretrained caches)
# ============================================================
echo "=== Phase 4: Thumbnails ==="

for SCENE in "${MOVIE4_SCENES[@]}"; do
  NPZ=$EXPORTS/pretrained/$SCENE/cache.npz
  OUT=$STATIC/packed/pretrained/$SCENE.packed
  if [ -f "$NPZ" ]; then
    echo "  Thumbnail $SCENE"
    $PACK --npz_path $NPZ --output $OUT --width 512 \
          --thumbnail --thumb_dir $STATIC/images/thumbs
  fi
done

echo "=== build_movie4.sh complete ==="
