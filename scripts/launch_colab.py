"""Run the match-tagger web UI on Google Colab (NVIDIA GPU) with a public share link.

The app is written for macOS: it decodes video with VideoToolbox and picks MPS or CPU.
This launcher adapts both at runtime without editing the app:
  - ffmpeg's "-hwaccel videotoolbox" is dropped on Linux (via scripts/evaluate_gar.py's patch)
  - "auto" device selection prefers CUDA when available
  - scoring uses one 8 s window per batch, so long matches fit in a T4's memory
then starts the same Gradio UI with share=True, which prints a https://xxxx.gradio.live link.

Usage (from the repo root on Colab):  python scripts/launch_colab.py [--extra-path /content/drive]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_gar as eg  # noqa: E402  (imports app/tdeed_runner and applies the Linux ffmpeg fix)

tr = eg.tr
_orig_pick_device = tr.pick_device


def _pick_device(name: str = "auto"):
    if name == "auto" and tr.torch.cuda.is_available():
        return tr.torch.device("cuda")
    return _orig_pick_device(name)


tr.pick_device = _pick_device

# The app scores two 8 s windows per batch, which runs a 15 GB T4 out of memory on long
# videos (seen in the padding vs context experiment). Score one window at a time instead.
_orig_compute_scores = tr.compute_scores


def _compute_scores(*args, **kwargs):
    kwargs.setdefault("batch_size", 1)
    return _orig_compute_scores(*args, **kwargs)


tr.compute_scores = _compute_scores

import ui  # noqa: E402  (app/ui.py, already on sys.path via evaluate_gar)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-path", action="append", default=[],
                    help="folder the UI may read videos from via the local-path box (e.g. /content/drive)")
    a = ap.parse_args()
    print(f"Model will run on: {tr.pick_device('auto')}")
    ui.build().queue(default_concurrency_limit=1).launch(
        share=True,
        allowed_paths=[str(tr.REPO_ROOT / "outputs"), *a.extra_path],
    )


if __name__ == "__main__":
    main()
