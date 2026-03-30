#!/bin/bash
#SBATCH -J selfevo_sky_all
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
BATCH="python website/selfevo/tools/run_inference_batch.py"
PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS=website/selfevo/exports
STATIC=website/selfevo/static

CKPT_G=training/logs/random_995_frcam/ckpts/checkpoint_11.pt
CKPT_Z=training/logs/zhenhuan_random_995_frcam/ckpts/checkpoint_11.pt
CKPT_T=training/logs/toystory_random_995_frcam/ckpts/checkpoint_11.pt
CKPT_D=training/logs/dragon_random_995_frcam/ckpts/checkpoint_3.pt

# ============================================================
# Phase 0: Delete old caches for all 45 target scenes
#   (so --skip_existing doesn't skip them on re-run)
# ============================================================
echo "=== Phase 0: Deleting old caches for target scenes ==="

for SCENE in \
    toystory-start toystory-run4 \
    toystory2-clip2 toystory2-clip3 toystory2-clip7 toystory2-clip10 toystory2-clip17 \
    zhenhuan-dance2 zhenhuan-street3 zhenhuan-1 zhenhuan-2 zhenhuan-3 \
    dragon-fly2 dragon-meet3 dragon-battle2 \
    dragon2-draw3 \
    wild-15416627_3840_2160_30fps wild-5039487-uhd_4096_2160_25fps \
    wild-5207050-uhd_3840_2160_25fps wild-6470850-2 \
    wild-6539141-hd_1920_1080_25fps2 wild-clip01 wild-clip_05 \
    wild-penguin wild-sport wild-tom1 \
    wild2-cat-piano wild2-cat-walk-right wild2-cat-walking \
    wild2-corgi-snow-2 wild2-corgi-snow-3 wild2-cow-walking \
    wild2-dog-rotation-vase wild2-kitten wild2-monkey wild2-sea-lion \
    davis-bear davis-blackswan davis-boat davis-breakdance davis-breakdance-flare \
    davis-car-roundabout davis-car-shadow davis-crossing davis-dance-twirl davis-dancing \
    davis-drift-chicane davis-drift-straight davis-drift-turn davis-hike \
    davis-horsejump-high davis-india davis-kid-football davis-kite-surf \
    davis-lab-coat davis-mallard-water davis-motocross-bumps davis-motorbike \
    davis-paragliding-launch davis-parkour davis-snowboard davis-soapbox \
    davis-stunt davis-tuk-tuk davis-walking; do
  rm -f $EXPORTS/pretrained/$SCENE/cache.npz
  rm -f $EXPORTS/selfevo/$SCENE/cache.npz
done

# ============================================================
# Phase 1: Generalization batch scenes (DAVIS + wild + wild2)
#   Pretrained and SelfEvo (game checkpoint)
# ============================================================

echo "=== Phase 1a: Pretrained DAVIS ==="
$BATCH \
  --input_root examples/DAVIS_480p_1of5 \
  --output_root $EXPORTS/pretrained \
  --prefix davis- \
  --mask_sky

echo "=== Phase 1b: Pretrained wild ==="
$BATCH \
  --video_dir examples/wild \
  --output_root $EXPORTS/pretrained \
  --prefix wild- \
  --target_fps 1 \
  --mask_sky

echo "=== Phase 1c: Pretrained wild2 ==="
$BATCH \
  --video_dir examples/wild2 \
  --output_root $EXPORTS/pretrained \
  --prefix wild2- \
  --target_fps 1 \
  --mask_sky

echo "=== Phase 1d: SelfEvo DAVIS ==="
$BATCH \
  --input_root examples/DAVIS_480p_1of5 \
  --output_root $EXPORTS/selfevo \
  --prefix davis- \
  --ckpt_path $CKPT_G \
  --mask_sky

echo "=== Phase 1e: SelfEvo wild ==="
$BATCH \
  --video_dir examples/wild \
  --output_root $EXPORTS/selfevo \
  --prefix wild- \
  --ckpt_path $CKPT_G \
  --target_fps 1 \
  --mask_sky

echo "=== Phase 1f: SelfEvo wild2 ==="
$BATCH \
  --video_dir examples/wild2 \
  --output_root $EXPORTS/selfevo \
  --prefix wild2- \
  --ckpt_path $CKPT_G \
  --target_fps 1 \
  --mask_sky

# ============================================================
# Phase 2: Adaptation batch scenes (toystory2 + dragon2)
# ============================================================

echo "=== Phase 2a: Pretrained dragon2 ==="
$BATCH \
  --video_dir examples/movie/dragon2 \
  --output_root $EXPORTS/pretrained \
  --prefix dragon2- \
  --target_fps 10 --frame_stride 10 \
  --mask_sky

echo "=== Phase 2b: Pretrained toystory2 ==="
$BATCH \
  --video_dir examples/movie/toystory2 \
  --output_root $EXPORTS/pretrained \
  --prefix toystory2- \
  --target_fps 10 --frame_stride 10 \
  --mask_sky

echo "=== Phase 2c: SelfEvo dragon2 ==="
$BATCH \
  --video_dir examples/movie/dragon2 \
  --output_root $EXPORTS/selfevo \
  --prefix dragon2- \
  --ckpt_path $CKPT_D \
  --target_fps 10 --frame_stride 10 \
  --mask_sky

echo "=== Phase 2d: SelfEvo toystory2 ==="
$BATCH \
  --video_dir examples/movie/toystory2 \
  --output_root $EXPORTS/selfevo \
  --prefix toystory2- \
  --ckpt_path $CKPT_T \
  --target_fps 10 --frame_stride 10 \
  --mask_sky

# ============================================================
# Phase 3: Adaptation single scenes (run_inference.py --mask_sky)
#   10 scenes × 2 versions = 20 invocations
# ============================================================

# --- Pretrained (HuggingFace weights) ---
echo "=== Phase 3a: Pretrained zhenhuan ==="
$INFER --video_path examples/movie/zhenhuan/dance2.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/zhenhuan-dance2/cache.npz
$INFER --video_path examples/movie/zhenhuan/street3.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/zhenhuan-street3/cache.npz
$INFER --image_folder examples/movie/zhenhuan/1 --mask_sky \
  --output $EXPORTS/pretrained/zhenhuan-1/cache.npz
$INFER --image_folder examples/movie/zhenhuan/2 --mask_sky \
  --output $EXPORTS/pretrained/zhenhuan-2/cache.npz
$INFER --image_folder examples/movie/zhenhuan/3 --mask_sky \
  --output $EXPORTS/pretrained/zhenhuan-3/cache.npz

echo "=== Phase 3b: Pretrained toystory ==="
$INFER --video_path examples/movie/toystory/start.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/toystory-start/cache.npz
$INFER --video_path examples/movie/toystory/run4.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/toystory-run4/cache.npz

echo "=== Phase 3c: Pretrained dragon ==="
$INFER --video_path examples/movie/dragon/fly2.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/dragon-fly2/cache.npz
$INFER --video_path examples/movie/dragon/meet3.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/dragon-meet3/cache.npz
$INFER --video_path examples/movie/dragon/battle2.mov --target_fps 10 --mask_sky \
  --output $EXPORTS/pretrained/dragon-battle2/cache.npz

# --- SelfEvo: zhenhuan checkpoint ---
echo "=== Phase 3d: SelfEvo zhenhuan ==="
$INFER --video_path examples/movie/zhenhuan/dance2.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-dance2/cache.npz
$INFER --video_path examples/movie/zhenhuan/street3.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-street3/cache.npz
$INFER --image_folder examples/movie/zhenhuan/1 --mask_sky \
  --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-1/cache.npz
$INFER --image_folder examples/movie/zhenhuan/2 --mask_sky \
  --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-2/cache.npz
$INFER --image_folder examples/movie/zhenhuan/3 --mask_sky \
  --ckpt_path $CKPT_Z --output $EXPORTS/selfevo/zhenhuan-3/cache.npz

# --- SelfEvo: toystory checkpoint ---
echo "=== Phase 3e: SelfEvo toystory ==="
$INFER --video_path examples/movie/toystory/start.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_T --output $EXPORTS/selfevo/toystory-start/cache.npz
$INFER --video_path examples/movie/toystory/run4.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_T --output $EXPORTS/selfevo/toystory-run4/cache.npz

# --- SelfEvo: dragon checkpoint ---
echo "=== Phase 3f: SelfEvo dragon ==="
$INFER --video_path examples/movie/dragon/fly2.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-fly2/cache.npz
$INFER --video_path examples/movie/dragon/meet3.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-meet3/cache.npz
$INFER --video_path examples/movie/dragon/battle2.mov --target_fps 10 --mask_sky \
  --ckpt_path $CKPT_D --output $EXPORTS/selfevo/dragon-battle2/cache.npz

# ============================================================
# Phase 4: Pack all 45 scenes
# ============================================================
echo "=== Phase 4: Packing all scenes ==="

ALL_SCENES=(
  toystory-start toystory-run4
  toystory2-clip2 toystory2-clip3 toystory2-clip7 toystory2-clip10 toystory2-clip17
  zhenhuan-dance2 zhenhuan-street3 zhenhuan-1 zhenhuan-2 zhenhuan-3
  dragon-fly2 dragon-meet3 dragon-battle2
  dragon2-draw3
  wild-15416627_3840_2160_30fps wild-5039487-uhd_4096_2160_25fps
  wild-5207050-uhd_3840_2160_25fps wild-6470850-2
  wild-6539141-hd_1920_1080_25fps2 wild-clip01 wild-clip_05
  wild-penguin wild-sport wild-tom1
  wild2-cat-piano wild2-cat-walk-right wild2-cat-walking
  wild2-corgi-snow-2 wild2-corgi-snow-3 wild2-cow-walking
  wild2-dog-rotation-vase wild2-kitten wild2-monkey wild2-sea-lion
  davis-bear davis-blackswan davis-boat davis-breakdance davis-breakdance-flare
  davis-car-roundabout davis-car-shadow davis-crossing davis-dance-twirl davis-dancing
  davis-drift-chicane davis-drift-straight davis-drift-turn davis-hike davis-horsejump-high
  davis-india davis-kid-football davis-kite-surf davis-lab-coat davis-mallard-water
  davis-motocross-bumps davis-motorbike davis-paragliding-launch davis-parkour
  davis-snowboard davis-soapbox davis-stunt davis-tuk-tuk davis-walking
)

for SCENE in "${ALL_SCENES[@]}"; do
  EXTRA_ARGS=""
  case "$SCENE" in
    dragon-fly2)    EXTRA_ARGS="--frame_start 0 --frame_end 7" ;;
    dragon-battle2) EXTRA_ARGS="--frame_start 0 --frame_end 9" ;;
  esac

  for VERSION in pretrained selfevo; do
    NPZ=$EXPORTS/$VERSION/$SCENE/cache.npz
    OUT=$STATIC/packed/$VERSION/$SCENE.packed
    if [ -f "$NPZ" ]; then
      $PACK --npz_path $NPZ --output $OUT --width 512 $EXTRA_ARGS
    else
      echo "WARNING: $VERSION/$SCENE — cache not found, skipping"
    fi
  done
done

# ============================================================
# Phase 5: Thumbnails (from pretrained caches)
# ============================================================
echo "=== Phase 5: Thumbnails ==="

for SCENE in "${ALL_SCENES[@]}"; do
  EXTRA_ARGS=""
  case "$SCENE" in
    dragon-fly2)    EXTRA_ARGS="--frame_start 0 --frame_end 7" ;;
    dragon-battle2) EXTRA_ARGS="--frame_start 0 --frame_end 9" ;;
  esac

  NPZ=$EXPORTS/pretrained/$SCENE/cache.npz
  OUT=$STATIC/packed/pretrained/$SCENE.packed
  if [ -f "$NPZ" ]; then
    $PACK --npz_path $NPZ --output $OUT --width 512 $EXTRA_ARGS \
          --thumbnail --thumb_dir $STATIC/images/thumbs
  fi
done

echo "=== build_all_sky.sh complete ==="
