#!/bin/bash
#SBATCH -J selfevo_repack
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH -o website/selfevo/logs/%j.out
#SBATCH -e website/selfevo/logs/%j.err

# No GPU needed — this only runs pack_scene.py (CPU only)

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

REPO=/home/nan.huang/code/vggt
cd $REPO
export PYTHONPATH=$REPO

PACK="python website/selfevo/tools/pack_scene.py"
EXPORTS=website/selfevo/exports
STATIC=website/selfevo/static

# All scenes currently in SCENE_CONFIG (including the new ones added)
ALL_SCENES=(
  # adaptation
  toystory-start toystory-run4
  toystory2-clip1 toystory2-clip2 toystory2-clip3 toystory2-clip4 toystory2-clip5
  toystory2-clip6 toystory2-clip7 toystory2-clip8 toystory2-clip9 toystory2-clip10
  toystory2-clip11 toystory2-clip12 toystory2-clip13 toystory2-clip14 toystory2-clip15
  toystory2-clip16 toystory2-clip17 toystory2-clip18 toystory2-clip19
  zhenhuan-dance2 zhenhuan-street3 zhenhuan-1 zhenhuan-2 zhenhuan-3
  dragon-fly2 dragon-meet3 dragon-battle2
  dragon2-clip1 dragon2-clip2 dragon2-clip3 dragon2-clip4 dragon2-clip5 dragon2-clip6
  dragon2-draw2 dragon2-draw3 dragon2-meet2 dragon2-battle3 dragon2-battle4
  # generalization — wild
  wild-15416627_3840_2160_30fps wild-4158842-uhd_3840_2160_24fps
  wild-5039487-uhd_4096_2160_25fps wild-5207050-uhd_3840_2160_25fps
  wild-6470850-2 wild-6539141-hd_1920_1080_25fps wild-6539141-hd_1920_1080_25fps2
  wild-clip01 wild-clip_05 wild-penguin wild-sport wild-tom1
  wild-drift-straight wild-horsejump-high wild-skating
  # generalization — wild2
  wild2-anemone wild2-boat-lake wild2-cat-piano wild2-cat-walk-left wild2-cat-walk-right
  wild2-cat-walking wild2-catepillar-1 wild2-corgi-snow wild2-corgi-snow-2 wild2-corgi-snow-3
  wild2-cow-walking wild2-dog-rotation-vase wild2-elephant wild2-horses-rotation
  wild2-kitten wild2-lady-birds wild2-lion wild2-monkey wild2-penguin
  wild2-pigeon wild2-pigeons wild2-sea-lion wild2-turtle
  # generalization — DAVIS
  davis-bear davis-blackswan davis-boat davis-breakdance davis-breakdance-flare
  davis-car-roundabout davis-car-shadow davis-crossing davis-dance-twirl davis-dancing
  davis-drift-chicane davis-drift-straight davis-drift-turn davis-hike davis-horsejump-high
  davis-india davis-kid-football davis-kite-surf davis-lab-coat davis-mallard-water
  davis-motocross-bumps davis-motorbike davis-paragliding-launch davis-parkour
  davis-snowboard davis-soapbox davis-stunt davis-tuk-tuk davis-walking
)

echo "=== Repacking ${#ALL_SCENES[@]} scenes with confidence masking fix ==="

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

echo "=== Repacking thumbnails (pretrained only) ==="
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

echo "=== repack_all.sh complete ==="
