#!/bin/bash
#SBATCH -J selfevo_wild15416
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

INFER="python website/selfevo/tools/run_inference_batch.py"
PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS=website/selfevo/exports
STATIC=website/selfevo/static
CKPT_G=training/logs/random_995_frcam/ckpts/checkpoint_11.pt
VIDEO=examples/wild/15416627_3840_2160_30fps.mp4
NAME=wild-15416627_3840_2160_30fps

echo "=== Pretrained: $NAME (fps=1 → ~55 frames) ==="
$INFER \
  --video_dir examples/wild \
  --output_root $EXPORTS/pretrained \
  --prefix wild- \
  --target_fps 1 \
  --mask_sky

echo "=== SelfEvo: $NAME ==="
$INFER \
  --video_dir examples/wild \
  --output_root $EXPORTS/selfevo \
  --prefix wild- \
  --ckpt_path $CKPT_G \
  --target_fps 1 \
  --mask_sky

echo "=== Pack + Thumbnail ==="
for VERSION in pretrained selfevo; do
  NPZ=$EXPORTS/$VERSION/$NAME/cache.npz
  OUT=$STATIC/packed/$VERSION/$NAME.packed
  [ -f "$NPZ" ] && $PACK --npz_path $NPZ --output $OUT --width 512
done

$PACK --npz_path $EXPORTS/pretrained/$NAME/cache.npz \
      --output $STATIC/packed/pretrained/$NAME.packed \
      --width 512 --thumbnail --thumb_dir $STATIC/images/thumbs

echo "=== Done ==="
