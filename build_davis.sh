#!/bin/bash
#SBATCH -J selfevo_davis
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
DAVIS=examples/DAVIS_480p_1of5
EXPORTS=website/selfevo/exports
STATIC=website/selfevo/static
CKPT_G=training/logs/random_995_frcam/ckpts/checkpoint_11.pt

# ============================================================
# Pretrained: all DAVIS scenes (model loaded once)
# ============================================================
echo "=== Pretrained inference: all DAVIS ==="
$BATCH \
  --input_root $DAVIS \
  --output_root $EXPORTS/pretrained \
  --prefix davis-

# ============================================================
# SelfEvo (game checkpoint): all DAVIS scenes (model loaded once)
# ============================================================
echo "=== SelfEvo inference: all DAVIS ==="
$BATCH \
  --input_root $DAVIS \
  --output_root $EXPORTS/selfevo \
  --prefix davis- \
  --ckpt_path $CKPT_G

# ============================================================
# Pack all DAVIS scenes
# ============================================================
echo "=== Packing all DAVIS scenes ==="
SCENES=$(ls $DAVIS)
for SCENE in $SCENES; do
  NAME="davis-${SCENE}"
  for VERSION in pretrained selfevo; do
    NPZ=$EXPORTS/$VERSION/$NAME/cache.npz
    OUT=$STATIC/packed/$VERSION/$NAME.packed
    if [ -f "$NPZ" ]; then
      $PACK --npz_path $NPZ --output $OUT --width 512
    else
      echo "SKIP $VERSION/$NAME — cache not found"
    fi
  done
done

# ============================================================
# Thumbnails (from pretrained cache)
# ============================================================
echo "=== Thumbnails ==="
for SCENE in $SCENES; do
  NAME="davis-${SCENE}"
  NPZ=$EXPORTS/pretrained/$NAME/cache.npz
  OUT=$STATIC/packed/pretrained/$NAME.packed
  if [ -f "$NPZ" ]; then
    $PACK --npz_path $NPZ --output $OUT --width 512 \
          --thumbnail --thumb_dir $STATIC/images/thumbs
  fi
done

echo "=== build_davis.sh complete ==="
