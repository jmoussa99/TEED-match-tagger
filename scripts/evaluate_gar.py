"""Evaluate the T-DEED tagger on labeled SoccerNet-GAR video clips.

Every GAR clip is 4.8 s long and its labeled action happens at ~2.43 s (the clip center).
For each clip we run the same scoring pipeline as the app (app/tdeed_runner.compute_scores),
take each class's highest score within +/- WINDOW_S seconds of the action time, and predict
the class with the highest score. Predictions are restricted to the 10 GAR classes by default
(T-DEED's extra classes DRIVE and BALL PLAYER BLOCK do not exist in GAR).

Usage (from this repo's root, after scripts/setup_tdeed.sh):
  .venv/bin/python scripts/evaluate_gar.py                       # every clip found in --videos
  .venv/bin/python scripts/evaluate_gar.py --per-class 20        # at most 20 clips per class
  .venv/bin/python scripts/evaluate_gar.py --limit 10            # quick smoke test

Outputs (outputs/gar_eval/): predictions.csv, per_class.csv, confusion.csv, metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
import tdeed_runner as tr  # noqa: E402

if sys.platform != "darwin":
    # The app decodes with macOS VideoToolbox ("-hwaccel videotoolbox"), which does not exist on
    # Linux (e.g. Google Colab). Re-use the app's decoder with that flag removed.
    import subprocess as _sp
    _orig_popen = _sp.Popen

    def _popen_without_videotoolbox(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "videotoolbox" in cmd:
            i = cmd.index("videotoolbox")
            cmd = cmd[: i - 1] + cmd[i + 1:]
        return _orig_popen(cmd, *args, **kwargs)

    tr.subprocess.Popen = _popen_without_videotoolbox

GAR_PROJECT = REPO_ROOT.parent / "sn-gar-2027"
ACTION_TIME_S = 2.43   # alignment.action_offset_seconds in the GAR video metadata
WINDOW_S = 1.0         # +/- 1 s, the same tolerance as the SoccerNet mAP@1 metric
OUT_DIR = REPO_ROOT / "outputs" / "gar_eval"


def load_labels(annotations: Path) -> dict[str, str]:
    data = json.load(open(annotations))["data"]
    return {s["id"]: s["labels"]["action"]["label"] for s in data}


def clip_id(path: Path, split: str) -> str:
    # clip_000123.mp4 -> valid_000123
    return f"{split}_{path.stem.split('_')[-1]}"


def predict_from_scores(scores: np.ndarray, classes: list[str]) -> tuple[str, float, dict[str, float]]:
    """scores: (N, 1 + 12) per-frame probabilities at tr.SAMPLED_FPS (column 0 = background)."""
    center = int(round(ACTION_TIME_S * tr.SAMPLED_FPS))
    half = int(round(WINDOW_S * tr.SAMPLED_FPS))
    lo, hi = max(0, center - half), min(len(scores), center + half + 1)
    window = scores[lo:hi, 1:]                      # drop background
    peak = window.max(axis=0) if len(window) else np.zeros(len(tr.CLASSES))
    per_class = {c: float(peak[i]) for i, c in enumerate(tr.CLASSES)}
    best = max(classes, key=lambda c: per_class[c])
    return best, per_class[best], per_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--annotations", type=Path, default=None,
                    help="GAR annotations JSON (default: ../sn-gar-2027/data/annotations_<split>.json)")
    ap.add_argument("--videos", type=Path, default=None,
                    help="folder of GAR clips (default: ../sn-gar-2027/data/videos/<split>)")
    ap.add_argument("--per-class", type=int, default=None, help="max clips per class")
    ap.add_argument("--limit", type=int, default=None, help="max clips in total")
    ap.add_argument("--all-12", action="store_true", help="allow DRIVE / BALL PLAYER BLOCK predictions")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="auto = cuda if available, else mps, else cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    annotations = a.annotations or GAR_PROJECT / "data" / f"annotations_{a.split}.json"
    videos_dir = a.videos or GAR_PROJECT / "data" / "videos" / a.split
    labels = load_labels(annotations)

    clips = sorted(videos_dir.glob("clip_*.mp4"))
    clips = [p for p in clips if clip_id(p, a.split) in labels]
    if not clips:
        sys.exit(f"No labeled clips found in {videos_dir}")
    if a.per_class:
        rng = random.Random(a.seed)
        by_class: dict[str, list[Path]] = {}
        for p in clips:
            by_class.setdefault(labels[clip_id(p, a.split)], []).append(p)
        clips = sorted(p for ps in by_class.values() for p in rng.sample(ps, min(a.per_class, len(ps))))
    if a.limit:
        clips = clips[: a.limit]
    print(f"{len(clips)} clips:", dict(Counter(labels[clip_id(p, a.split)] for p in clips).most_common()))

    classes = tr.CLASSES if a.all_12 else tr.GAR_CLASSES
    device = a.device
    if device == "auto" and tr.torch.cuda.is_available():
        device = "cuda"
    model = tr.get_model(device)
    print(f"Model on {tr.pick_device(device)}")
    rows, t0 = [], time.perf_counter()
    for i, p in enumerate(clips, 1):
        scores = tr.compute_scores(p, model=model)
        pred, conf, per_class = predict_from_scores(scores, classes)
        cid = clip_id(p, a.split)
        rows.append({"id": cid, "label": labels[cid], "pred": pred, "confidence": round(conf, 4),
                     **{f"score_{c}": round(v, 4) for c, v in per_class.items()}})
        if i % 10 == 0 or i == len(clips):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(clips)} clips · {el / i:.1f} s per clip · ETA {el / i * (len(clips) - i) / 60:.1f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "predictions.csv", index=False)

    present = [c for c in tr.GAR_CLASSES if (df.label == c).any()]
    recall = {c: float((df[df.label == c].pred == c).mean()) for c in present}
    precision = {c: float((df[df.pred == c].label == c).mean()) if (df.pred == c).any() else float("nan")
                 for c in present}
    per_class = pd.DataFrame({"class": present,
                              "n_clips": [int((df.label == c).sum()) for c in present],
                              "recall": [round(recall[c], 3) for c in present],
                              "precision": [round(precision[c], 3) for c in present]})
    per_class.to_csv(OUT_DIR / "per_class.csv", index=False)
    pd.crosstab(df.label, df.pred).to_csv(OUT_DIR / "confusion.csv")
    metrics = {"split": a.split, "n_clips": len(df), "classes_present": len(present),
               "prediction_classes": "all 12" if a.all_12 else "GAR 10",
               "window_seconds": WINDOW_S, "action_time_seconds": ACTION_TIME_S,
               "balanced_accuracy": 100 * float(np.mean(list(recall.values()))),
               "accuracy": 100 * float((df.label == df.pred).mean()),
               "seconds_per_clip": round((time.perf_counter() - t0) / len(df), 2)}
    json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2)

    print(json.dumps(metrics, indent=2))
    print(per_class.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
