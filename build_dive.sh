#!/bin/bash
#SBATCH -J selfevo_dive
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_dive.out
#SBATCH -e /home/nan.huang/code/page/Self-Evo.github.io/logs/%j_dive.err

set -e

source /home/ed25519/miniconda3/etc/profile.d/conda.sh
conda activate robust_vggt

SITE=/home/nan.huang/code/page/Self-Evo.github.io
VGGT=/home/nan.huang/code/vggt
export PYTHONPATH=$VGGT

INFER="python $SITE/tools/run_inference.py"
PACK="python $SITE/tools/pack_scene.py"
EXPORTS=$SITE/exports
STATIC=$SITE/static
CKPT_G=/home/nan.huang/code/vggt-exp/training/logs/random_995_frcam/ckpts/checkpoint_11.pt

# ============================================================
# Prepare 5-frame subset: 0000, 0080, 0160, 0240, 0320
# ============================================================
DIVE_5=$SITE/examples/dive_5
mkdir -p $DIVE_5
cp $SITE/examples/dive/0000.jpg $DIVE_5/
cp $SITE/examples/dive/0080.jpg $DIVE_5/
cp $SITE/examples/dive/0160.jpg $DIVE_5/
cp $SITE/examples/dive/0240.jpg $DIVE_5/
cp $SITE/examples/dive/0320.jpg $DIVE_5/

# ============================================================
# Inference on 5 frames
# ============================================================
echo "=== Pretrained: dive (5 frames) ==="
$INFER --image_folder $DIVE_5 \
       --output $EXPORTS/pretrained/wild-dive/cache.npz

echo "=== SelfEvo: dive (5 frames) ==="
$INFER --image_folder $DIVE_5 \
       --ckpt_path $CKPT_G \
       --output $EXPORTS/selfevo/wild-dive/cache.npz

# ============================================================
# Pack
# ============================================================
echo "=== Packing ==="
$PACK --npz_path $EXPORTS/pretrained/wild-dive/cache.npz \
      --output $STATIC/packed/pretrained/wild-dive.packed \
      --width 512 --thumbnail --thumb_dir $STATIC/images/thumbs

$PACK --npz_path $EXPORTS/selfevo/wild-dive/cache.npz \
      --output $STATIC/packed/selfevo/wild-dive.packed \
      --width 512

# Cleanup
rm -rf $DIVE_5

echo "=== build_dive.sh complete ==="
