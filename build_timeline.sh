#!/bin/bash
#SBATCH -J selfevo_timeline
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_timeline.out
#SBATCH -e /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_timeline.err

# Build script: SelfEvo Timeline Demo
# Runs inference for pretrained + checkpoints 0-12 on all Interactive Results scenes,
# then packs the results into static/packed/timeline/{version}/{scene}.packed.
#
# Usage (interactive):  bash build_timeline.sh
# Usage (SLURM):        sbatch build_timeline.sh
#
# Estimated time:  ~6-8 hours (14 model loads x ~2 min x 15 new scenes)
# GPU memory:       ~24 GB (VGGT-1B fp16)

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

REPO=/home/nan.huang/code/vggt-exp
cd $REPO
export PYTHONPATH=$REPO

WEBSITE=/home/nan.huang/code/page/Self-Evo.github.io
INFER="python $WEBSITE/tools/run_inference.py"
PACK="python $WEBSITE/tools/pack_scene.py"
EXPORTS=$WEBSITE/exports
STATIC=$WEBSITE/static
CKPT_DIR=$REPO/training/logs/random_995_frcam/ckpts

mkdir -p $WEBSITE/logs

# ── Scene list: name → input image folder ─────────────────────────────
# Format: "scene-name:path/to/frames"
# All use --image_folder (pre-extracted frames) and --mask_sky.
SCENES=(
  "davis-kid-football:$REPO/examples/DAVIS_480p_1of5/kid-football"
"wild-5207050-uhd_3840_2160_25fps:$REPO/examples/wild/5207050-uhd_3840_2160_25fps_frames"
  "wild-6539141-hd_1920_1080_25fps2:$REPO/examples/wild/6539141-hd_1920_1080_25fps2_frames"
  "wild-clip_05:$REPO/examples/wild/clip_05_frames"
  "wild-penguin:$REPO/examples/wild/penguin_frames"
  "wild-sport:$REPO/examples/wild/sport_frames"
  "wild-tom1:$REPO/examples/wild/tom1_frames"
  "history-1:$REPO/examples/movie/history/1"
  "wild2-corgi-snow-2:$REPO/examples/wild2/corgi-snow-2_frames"
  "davis-car-roundabout:$REPO/examples/DAVIS_480p_1of5/car-roundabout"
  "davis-drift-straight:$REPO/examples/DAVIS_480p_1of5/drift-straight"
  "davis-drift-turn:$REPO/examples/DAVIS_480p_1of5/drift-turn"
  "davis-snowboard:$REPO/examples/DAVIS_480p_1of5/snowboard"
  "davis-blackswan:$REPO/examples/DAVIS_480p_1of5/blackswan"
  "davis-walking:$REPO/examples/DAVIS_480p_1of5/walking"
)

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Inference
# Batched by checkpoint so each model loads only once per version.
# Skips scenes whose cache.npz already exists.
# ══════════════════════════════════════════════════════════════════════════

run_inference_for_version() {
  local VERSION_DIR=$1   # e.g. "timeline_pretrain"
  local CKPT_ARG=$2      # e.g. "" or "--ckpt_path /path/to/ckpt.pt"

  echo ""
  echo "=== Inference: $VERSION_DIR ==="

  local any_needed=0
  for entry in "${SCENES[@]}"; do
    local SCENE="${entry%%:*}"
    local NPZ="$EXPORTS/$VERSION_DIR/$SCENE/cache.npz"
    if [ ! -f "$NPZ" ]; then
      any_needed=1
      break
    fi
  done

  if [ "$any_needed" -eq 0 ]; then
    echo "  All scenes already cached, skipping model load."
    return
  fi

  for entry in "${SCENES[@]}"; do
    local SCENE="${entry%%:*}"
    local FOLDER="${entry##*:}"
    local NPZ="$EXPORTS/$VERSION_DIR/$SCENE/cache.npz"

    if [ -f "$NPZ" ]; then
      echo "  SKIP: $SCENE (cache exists)"
      continue
    fi

    if [ ! -d "$FOLDER" ]; then
      echo "  WARNING: input dir not found: $FOLDER — skipping $SCENE"
      continue
    fi

    echo "  Running: $SCENE"
    mkdir -p "$(dirname $NPZ)"
    $INFER --image_folder "$FOLDER" --mask_sky $CKPT_ARG \
      --output "$NPZ"
  done
}

# Pretrained (no checkpoint; loads HuggingFace weights)
run_inference_for_version timeline_pretrain ""

# Checkpoints 0-12
for CKPT_IDX in $(seq 0 12); do
  CKPT_NAME=$(printf "ckpt_%02d" $CKPT_IDX)
  CKPT_FILE=$CKPT_DIR/checkpoint_${CKPT_IDX}.pt
  if [ ! -f "$CKPT_FILE" ]; then
    echo "WARNING: $CKPT_FILE not found, skipping"
    continue
  fi
  run_inference_for_version "timeline_${CKPT_NAME}" "--ckpt_path $CKPT_FILE"
done

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Pack
# Converts .npz caches to .packed binaries for the WebGL viewer.
# Skips scenes whose .packed already exists.
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Packing ==="

pack_version() {
  local VERSION_DIR=$1   # e.g. "timeline_pretrain"
  local VERSION_KEY=$2   # subdirectory under static/packed/timeline/

  for entry in "${SCENES[@]}"; do
    local SCENE="${entry%%:*}"
    local NPZ="$EXPORTS/$VERSION_DIR/$SCENE/cache.npz"
    local OUT="$STATIC/packed/timeline/$VERSION_KEY/$SCENE.packed"

    if [ -f "$OUT" ]; then
      echo "  SKIP: timeline/$VERSION_KEY/$SCENE.packed (exists)"
      continue
    fi

    if [ ! -f "$NPZ" ]; then
      echo "  WARNING: $NPZ not found, skipping"
      continue
    fi

    mkdir -p "$(dirname $OUT)"
    $PACK --npz_path "$NPZ" --output "$OUT" --width 512
    echo "  packed: timeline/$VERSION_KEY/$SCENE.packed"
  done
}

pack_version timeline_pretrain pretrain

for CKPT_IDX in $(seq 0 12); do
  CKPT_NAME=$(printf "ckpt_%02d" $CKPT_IDX)
  pack_version "timeline_${CKPT_NAME}" "$CKPT_NAME"
done

echo ""
echo "=== Done! ==="
echo "Timeline packed files written to: $STATIC/packed/timeline/"
ls -lh $STATIC/packed/timeline/pretrain/ | tail -5
