#!/bin/bash
#SBATCH -J selfevo_zhenhuan_new
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

echo "=== Pretrained: zhenhuan 1/2/3 ==="
$INFER --image_folder examples/movie/zhenhuan/1 --output $EXPORTS/pretrained/zhenhuan-1/cache.npz
$INFER --image_folder examples/movie/zhenhuan/2 --output $EXPORTS/pretrained/zhenhuan-2/cache.npz
$INFER --image_folder examples/movie/zhenhuan/3 --output $EXPORTS/pretrained/zhenhuan-3/cache.npz

echo "=== SelfEvo: zhenhuan 1/2/3 ==="
$INFER --image_folder examples/movie/zhenhuan/1 --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-1/cache.npz
$INFER --image_folder examples/movie/zhenhuan/2 --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-2/cache.npz
$INFER --image_folder examples/movie/zhenhuan/3 --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-3/cache.npz

echo "=== Packing ==="
for SCENE in zhenhuan-1 zhenhuan-2 zhenhuan-3; do
  for VERSION in pretrained selfevo; do
    NPZ=$EXPORTS/$VERSION/$SCENE/cache.npz
    OUT=$STATIC/packed/$VERSION/$SCENE.packed
    if [ -f "$NPZ" ]; then
      $PACK --npz_path $NPZ --output $OUT --width 512
    fi
  done
done

echo "=== Thumbnails ==="
for SCENE in zhenhuan-1 zhenhuan-2 zhenhuan-3; do
  NPZ=$EXPORTS/pretrained/$SCENE/cache.npz
  OUT=$STATIC/packed/pretrained/$SCENE.packed
  if [ -f "$NPZ" ]; then
    $PACK --npz_path $NPZ --output $OUT --thumbnail --thumb_dir $STATIC/images/thumbs
  fi
done

echo "=== Done ==="
