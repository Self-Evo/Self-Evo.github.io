#!/bin/bash
#SBATCH -J selfevo_gen
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
CKPT_G=training/logs/random_995_frcam/ckpts/checkpoint_11.pt

# ============================================================
# PRETRAINED — load model once per source
# ============================================================
echo "=== Pretrained: DAVIS (image folders) ==="
$BATCH \
  --input_root examples/DAVIS_480p_1of5 \
  --output_root $EXPORTS/pretrained \
  --prefix davis- \
  --mask_sky

echo "=== Pretrained: wild (videos @ 1 fps) ==="
$BATCH \
  --video_dir examples/wild \
  --output_root $EXPORTS/pretrained \
  --prefix wild- \
  --target_fps 1 \
  --mask_sky

echo "=== Pretrained: wild2 (videos @ 1 fps) ==="
$BATCH \
  --video_dir examples/wild2 \
  --output_root $EXPORTS/pretrained \
  --prefix wild2- \
  --target_fps 1 \
  --mask_sky

# ============================================================
# SELFEVO — load model once per source
# ============================================================
echo "=== SelfEvo: DAVIS (image folders) ==="
$BATCH \
  --input_root examples/DAVIS_480p_1of5 \
  --output_root $EXPORTS/selfevo \
  --prefix davis- \
  --ckpt_path $CKPT_G \
  --mask_sky

echo "=== SelfEvo: wild (videos @ 1 fps) ==="
$BATCH \
  --video_dir examples/wild \
  --output_root $EXPORTS/selfevo \
  --prefix wild- \
  --ckpt_path $CKPT_G \
  --target_fps 1 \
  --mask_sky

echo "=== SelfEvo: wild2 (videos @ 1 fps) ==="
$BATCH \
  --video_dir examples/wild2 \
  --output_root $EXPORTS/selfevo \
  --prefix wild2- \
  --ckpt_path $CKPT_G \
  --target_fps 1 \
  --mask_sky

# ============================================================
# Pack all scenes
# ============================================================
echo "=== Packing ==="
pack_dir() {
  local VERSION=$1
  local PREFIX=$2
  for NPZ in $EXPORTS/$VERSION/${PREFIX}*/cache.npz; do
    SCENE=$(basename $(dirname $NPZ))
    OUT=$STATIC/packed/$VERSION/$SCENE.packed
    if [ ! -f "$OUT" ] || [ "$NPZ" -nt "$OUT" ]; then
      python website/selfevo/tools/pack_scene.py \
        --npz_path $NPZ --output $OUT --width 512
    fi
  done
}

for VERSION in pretrained selfevo; do
  pack_dir $VERSION davis-
  pack_dir $VERSION wild-
  pack_dir $VERSION wild2-
done

# ============================================================
# Thumbnails (from pretrained, run after packing so we don't overwrite)
# ============================================================
echo "=== Thumbnails ==="
for PREFIX in davis- wild- wild2-; do
  for NPZ in $EXPORTS/pretrained/${PREFIX}*/cache.npz; do
    SCENE=$(basename $(dirname $NPZ))
    OUT=$STATIC/packed/pretrained/$SCENE.packed
    if [ -f "$NPZ" ]; then
      python website/selfevo/tools/pack_scene.py \
        --npz_path $NPZ --output $OUT --width 512 \
        --thumbnail --thumb_dir $STATIC/images/thumbs
    fi
  done
done

echo "=== build_generalization.sh complete ==="
