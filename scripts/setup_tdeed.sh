#!/usr/bin/env bash
# Set up T-DEED ball action spotting for the full-match tagging app (app/).
# Run from the repository root: bash scripts/setup_tdeed.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TDEED_DIR="private/T-DEED"
TDEED_COMMIT="d953b4de8a2ad941c565e1683c5b07255cbe644e"
MODEL_NAME="SoccerNetBall_challenge2"
# File id of SoccerNetBall/SoccerNetBall_challenge2/checkpoint_best.pt in the
# checkpoint folder linked from the T-DEED README.
CHECKPOINT_ID="1uCWam8EhAXt3z6QQAfhRfzHITUcml438"
PYTHON_BIN=".venv/bin/python"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required: brew install ffmpeg" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || uv venv --python 3.12 .venv

if [[ ! -d "$TDEED_DIR/.git" ]]; then
  git clone https://github.com/arturxe2/T-DEED "$TDEED_DIR"
fi
git -C "$TDEED_DIR" fetch -q origin
git -C "$TDEED_DIR" checkout -q "$TDEED_COMMIT"

# T-DEED hardcodes CUDA in a few places; the patch makes it device-agnostic.
if git -C "$TDEED_DIR" apply --check "$PWD/app/patches/tdeed_mps.patch" 2>/dev/null; then
  git -C "$TDEED_DIR" apply "$PWD/app/patches/tdeed_mps.patch"
  echo "Applied app/patches/tdeed_mps.patch"
else
  echo "Patch already applied (or tree modified); skipping"
fi

# T-DEED's requirements.txt pins Linux CUDA wheels; use this repository's instead.
uv pip install --python "$PYTHON_BIN" -r requirements.txt

CKPT_DIR="$TDEED_DIR/checkpoints/SoccerNetBall/$MODEL_NAME"
mkdir -p "$CKPT_DIR"
if [[ ! -s "$CKPT_DIR/checkpoint_best.pt" ]]; then
  .venv/bin/gdown "$CHECKPOINT_ID" -O "$CKPT_DIR/checkpoint_best.pt"
fi
ls -l "$CKPT_DIR/checkpoint_best.pt"
echo "T-DEED ready. Run: $PYTHON_BIN app/ui.py"
