"""Save T-DEED's full per-frame scores and a strip of video frames for every GAR clip.

For error analysis: the evaluation (scripts/evaluate_gar.py) keeps only each class's peak score
within +/-1 s of the action. Here we keep everything, so we can check whether the right class
fires at a different time (timing offset) and look at what the camera shows.

Usage:
  .venv/bin/python scripts/dump_gar_scores.py --clips DIR [--sheets]
Outputs (outputs/gar_dump/):
  scores.npz   one array per clip id, shape (frames at 12.5 fps, 1 + 12 classes), col 0 = background
  labels.csv   id, label, action_offset_s
  sheets/<id>.jpg  (with --sheets) 7 frames from -1.5 s to +1.5 s around the action
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_gar as eg  # noqa: E402  (also applies the Linux ffmpeg fix)

tr = eg.tr
OUT_DIR = eg.REPO_ROOT / "outputs" / "gar_dump"
SHEET_OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


def contact_sheet(video: Path, action_s: float, out: Path, width: int = 256) -> None:
    from PIL import Image, ImageDraw
    tiles = []
    for off in SHEET_OFFSETS:
        t = max(0.0, action_s + off)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                              "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                             capture_output=True, check=True).stdout
        img = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
        ImageDraw.Draw(img).text((6, 6), f"{off:+.1f}s", fill=(255, 255, 0))
        tiles.append(img)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, im in enumerate(tiles):
        sheet.paste(im, (i * w, 0))
    sheet.save(out, quality=80)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True, help="folder with clip_*.mp4 + clip_*.json")
    ap.add_argument("--sheets", action="store_true", help="also save a frame strip per clip")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    a = ap.parse_args()

    rows = []
    for js in sorted(a.clips.glob("clip_*.json")):
        mp4 = js.with_suffix(".mp4")
        if mp4.is_file():
            j = json.loads(js.read_text())
            rows.append({"id": j["id"], "label": j["labels"]["action"]["label"], "path": mp4,
                         "action_offset_s": j["metadata"]["conversion"]["alignment"]["action_offset_seconds"]})
    if not rows:
        sys.exit(f"No clip_*.mp4 + clip_*.json pairs in {a.clips}")
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="path").to_csv(OUT_DIR / "labels.csv", index=False)

    if a.sheets:
        (OUT_DIR / "sheets").mkdir(exist_ok=True)
        for r in df.itertuples():
            out = OUT_DIR / "sheets" / f"{r.id}.jpg"
            if not out.is_file():
                contact_sheet(r.path, r.action_offset_s, out)
        print(f"{len(df)} frame strips saved", flush=True)

    device = "cuda" if a.device == "auto" and tr.torch.cuda.is_available() else a.device
    model = tr.get_model(device)
    print(f"Model on {tr.pick_device(device)}; {len(df)} clips")
    scores, t0 = {}, time.perf_counter()
    for i, r in enumerate(df.itertuples(), 1):
        scores[r.id] = tr.compute_scores(r.path, model=model).astype(np.float32)
        if i % 25 == 0 or i == len(df):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(df)} · ETA {el / i * (len(df) - i) / 60:.1f} min", flush=True)
    np.savez_compressed(OUT_DIR / "scores.npz", **scores)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
