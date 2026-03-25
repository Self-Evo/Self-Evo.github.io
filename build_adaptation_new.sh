#!/bin/bash
#SBATCH -J selfevo_adapt_new
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

BATCH="python website/selfevo/tools/run_inference_batch.py"
PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS=website/selfevo/exports
STATIC=website/selfevo/static

CKPT_DRAGON=training/logs/dragon_random_995_frcam/ckpts/checkpoint_3.pt
CKPT_TOYSTORY=training/logs/toystory_random_995_frcam/ckpts/checkpoint_11.pt

# ============================================================
# Pretrained: dragon2 + toystory2 (model loaded once)
# ============================================================
echo "=== Pretrained: dragon2 + toystory2 ==="
$BATCH \
  --video_dir examples/movie/dragon2 \
  --output_root $EXPORTS/pretrained \
  --prefix dragon2- \
  --target_fps 10 \
  --mask_sky

$BATCH \
  --video_dir examples/movie/toystory2 \
  --output_root $EXPORTS/pretrained \
  --prefix toystory2- \
  --target_fps 10 \
  --mask_sky

# ============================================================
# SelfEvo: dragon2 (dragon checkpoint)
# ============================================================
echo "=== SelfEvo: dragon2 ==="
$BATCH \
  --video_dir examples/movie/dragon2 \
  --output_root $EXPORTS/selfevo \
  --prefix dragon2- \
  --ckpt_path $CKPT_DRAGON \
  --target_fps 10 \
  --mask_sky

# ============================================================
# SelfEvo: toystory2 (toystory checkpoint)
# ============================================================
echo "=== SelfEvo: toystory2 ==="
$BATCH \
  --video_dir examples/movie/toystory2 \
  --output_root $EXPORTS/selfevo \
  --prefix toystory2- \
  --ckpt_path $CKPT_TOYSTORY \
  --target_fps 10 \
  --mask_sky

# ============================================================
# Pack all new scenes
# ============================================================
echo "=== Packing ==="
for PREFIX in dragon2- toystory2-; do
  for VERSION in pretrained selfevo; do
    for NPZ in $EXPORTS/$VERSION/${PREFIX}*/cache.npz; do
      [ -f "$NPZ" ] || continue
      SCENE=$(basename $(dirname $NPZ))
      OUT=$STATIC/packed/$VERSION/$SCENE.packed
      $PACK --npz_path $NPZ --output $OUT --width 512
    done
  done
done

# ============================================================
# Thumbnails
# ============================================================
echo "=== Thumbnails ==="
for PREFIX in dragon2- toystory2-; do
  for NPZ in $EXPORTS/pretrained/${PREFIX}*/cache.npz; do
    [ -f "$NPZ" ] || continue
    SCENE=$(basename $(dirname $NPZ))
    OUT=$STATIC/packed/pretrained/$SCENE.packed
    $PACK --npz_path $NPZ --output $OUT --width 512 \
          --thumbnail --thumb_dir $STATIC/images/thumbs
  done
done

echo "=== build_adaptation_new.sh complete ==="
