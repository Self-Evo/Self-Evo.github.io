#!/bin/bash
#SBATCH -J selfevo_web         # job name
#SBATCH -p all                 # partition
#SBATCH -N 1                   # 1 node
#SBATCH --gres=gpu:1           # 1 GPU is enough for inference
#SBATCH -o website/selfevo/logs/%j.out
#SBATCH -e website/selfevo/logs/%j.err

set -e  # exit on any error

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

REPO=/home/nan.huang/code/vggt
cd $REPO

export PYTHONPATH=$REPO

INFER="python website/selfevo/tools/run_inference.py"
PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS="website/selfevo/exports"
STATIC="website/selfevo/static"
THUMBS="$STATIC/images/thumbs"

mkdir -p $THUMBS

# ============================================================
# PRETRAINED — official facebook/VGGT-1B (no --ckpt_path)
# ============================================================
echo "========== PRETRAINED: adaptation scenes =========="

$INFER --video_path examples/movie/zhenhuan/dance2.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/zhenhuan-dance2/cache.npz

$INFER --video_path examples/movie/zhenhuan/street3.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/zhenhuan-street3/cache.npz

$INFER --video_path examples/movie/toystory/start.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/toystory-start/cache.npz

$INFER --video_path examples/movie/toystory/run4.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/toystory-run4/cache.npz

$INFER --video_path examples/movie/dragon/fly2.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/dragon-fly2/cache.npz

$INFER --video_path examples/movie/dragon/meet3.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/dragon-meet3/cache.npz

$INFER --video_path examples/movie/dragon/battle2.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/dragon-battle2/cache.npz

echo "========== PRETRAINED: generalization scenes =========="

$INFER --image_folder examples/movie/history/1  \
       --output $EXPORTS/pretrained/history-1/cache.npz

$INFER --video_path examples/movie/tom/1.mov  --target_fps 10 \
       --output $EXPORTS/pretrained/tom-1/cache.npz

$INFER --video_path examples/wild/dive20.mp4  --target_fps 10 \
       --output $EXPORTS/pretrained/wild-dive20/cache.npz

$INFER --video_path examples/wild/sport.mp4  --target_fps 10 \
       --output $EXPORTS/pretrained/wild-sport/cache.npz

$INFER --video_path examples/wild/penguin.mp4  --target_fps 10 \
       --output $EXPORTS/pretrained/wild-penguin/cache.npz

# ============================================================
# SELFEVO — fine-tuned checkpoints
# ============================================================
echo "========== SELFEVO: adaptation — zhenhuan =========="
CKPT_Z=training/logs/zhenhuan_random_995_frcam/ckpts/checkpoint_11.pt

$INFER --video_path examples/movie/zhenhuan/dance2.mov  --target_fps 10 \
       --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-dance2/cache.npz

$INFER --video_path examples/movie/zhenhuan/street3.mov  --target_fps 10 \
       --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-street3/cache.npz

echo "========== SELFEVO: adaptation — toystory =========="
CKPT_T=training/logs/toystory_random_995_frcam/ckpts/checkpoint_11.pt

$INFER --video_path examples/movie/toystory/start.mov  --target_fps 10 \
       --ckpt_path $CKPT_T --output $EXPORTS/selfevo/toystory-start/cache.npz

$INFER --video_path examples/movie/toystory/run4.mov  --target_fps 10 \
       --ckpt_path $CKPT_T --output $EXPORTS/selfevo/toystory-run4/cache.npz

echo "========== SELFEVO: adaptation — dragon =========="
CKPT_D=training/logs/dragon_random_995_frcam/ckpts/checkpoint_3.pt

$INFER --video_path examples/movie/dragon/fly2.mov  --target_fps 10 \
       --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-fly2/cache.npz

$INFER --video_path examples/movie/dragon/meet3.mov  --target_fps 10 \
       --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-meet3/cache.npz

$INFER --video_path examples/movie/dragon/battle2.mov  --target_fps 10 \
       --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-battle2/cache.npz

echo "========== SELFEVO: generalization (game checkpoint) =========="
CKPT_G=training/logs/random_995_frcam/ckpts/checkpoint_11.pt

$INFER --image_folder examples/movie/history/1 \
       --ckpt_path $CKPT_G --output $EXPORTS/selfevo/history-1/cache.npz

$INFER --video_path examples/movie/tom/1.mov  --target_fps 10 \
       --ckpt_path $CKPT_G --output $EXPORTS/selfevo/tom-1/cache.npz

$INFER --video_path examples/wild/dive20.mp4  --target_fps 10 \
       --ckpt_path $CKPT_G --output $EXPORTS/selfevo/wild-dive20/cache.npz

$INFER --video_path examples/wild/sport.mp4  --target_fps 10 \
       --ckpt_path $CKPT_G --output $EXPORTS/selfevo/wild-sport/cache.npz

$INFER --video_path examples/wild/penguin.mp4  --target_fps 10 \
       --ckpt_path $CKPT_G --output $EXPORTS/selfevo/wild-penguin/cache.npz

# ============================================================
# PACK all scenes → .packed
# ============================================================
echo "========== Packing all scenes =========="
$PACK \
  --config website/selfevo/scene_list.json \
  --output_dir $STATIC/packed \
  --width 512

# ============================================================
# THUMBNAILS — extract first frame from each pretrained cache
# ============================================================
echo "========== Generating thumbnails =========="
SCENES=(
  zhenhuan-dance2 zhenhuan-street3
  toystory-start toystory-run4
  dragon-fly2 dragon-meet3 dragon-battle2
  history-1 tom-1
  wild-dive20 wild-sport wild-penguin
)

for SCENE in "${SCENES[@]}"; do
  NPZ=$EXPORTS/pretrained/$SCENE/cache.npz
  PACKED=$STATIC/packed/pretrained/$SCENE.packed
  if [ -f "$NPZ" ]; then
    $PACK --npz_path $NPZ --output $PACKED --thumbnail --thumb_dir $THUMBS
  else
    echo "[thumb] SKIP $SCENE — cache not found at $NPZ"
  fi
done

echo "========== build_website.sh complete =========="
