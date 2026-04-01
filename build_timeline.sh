#!/bin/bash
#SBATCH -J selfevo_timeline
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_timeline.out
#SBATCH -e /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_timeline.err

# Build script: SelfEvo Timeline Demo
# Runs inference for pretrained + checkpoints 0-12 on 3 target scenes,
# then packs the results into static/packed/timeline/{version}/{scene}.packed.
#
# Usage (interactive):  bash build_timeline.sh
# Usage (SLURM):        sbatch build_timeline.sh
#
# Estimated time:  ~3-4 hours (14 model loads x ~2 min inference x 3 scenes)
# GPU memory:       ~24 GB (VGGT-1B fp16)

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

# vggt-exp has both the Python module and all training artifacts + input examples
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

# ── Target scene ──────────────────────────────────────────────────────────
DAVIS_SCENE=davis-kid-football
DAVIS_DIR=$REPO/examples/DAVIS_480p_1of5/kid-football

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Inference
# Batched by checkpoint so each 16GB model loads only once per version.
# ══════════════════════════════════════════════════════════════════════════

run_inference_for_version() {
  local VERSION_DIR=$1   # e.g. "timeline_pretrain" or "timeline_ckpt_00"
  local CKPT_ARG=$2      # e.g. "" or "--ckpt_path /path/to/ckpt.pt"

  echo ""
  echo "=== Inference: $VERSION_DIR ==="

  $INFER --image_folder $DAVIS_DIR --mask_sky $CKPT_ARG \
    --output $EXPORTS/$VERSION_DIR/$DAVIS_SCENE/cache.npz
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
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Packing ==="

pack_version() {
  local VERSION_DIR=$1   # e.g. "timeline_pretrain"
  local VERSION_KEY=$2   # e.g. "pretrain"  (subdirectory under static/packed/timeline/)

  for SCENE in $DAVIS_SCENE; do
    NPZ=$EXPORTS/$VERSION_DIR/$SCENE/cache.npz
    OUT=$STATIC/packed/timeline/$VERSION_KEY/$SCENE.packed
    if [ -f "$NPZ" ]; then
      mkdir -p "$(dirname $OUT)"
      $PACK --npz_path $NPZ --output $OUT --width 512
      echo "  packed: timeline/$VERSION_KEY/$SCENE.packed"
    else
      echo "  WARNING: $NPZ not found, skipping"
    fi
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
ls -lh $STATIC/packed/timeline/
