"""Does padding hurt T-DEED? Compare the same GAR events with and without real context.

GAR clips are 4.8 s with the action at ~2.43 s, but T-DEED looks at 8 s windows, so a lone clip
is padded with blank frames. Consecutive GAR clips from the same game overlap in the source
match video, so we stitch them back into continuous footage (frame-exact, using each clip's
source frame indices) and score every event twice:

  padded  : the lone 4.8 s clip, exactly as scripts/evaluate_gar.py does
  context : the stitched continuous segment that contains the clip

Same model, same events, same decision rule (best GAR class within +/-1 s of the action).

Input: a folder of GAR clips with their sidecar JSON files (clip_XXXXXX.mp4 + clip_XXXXXX.json,
as stored in the dataset's video shards).

Usage:
  .venv/bin/python scripts/evaluate_gar_context.py --clips DIR
  .venv/bin/python scripts/evaluate_gar_context.py --clips DIR --stitch-only     # just build/check videos
Outputs (outputs/gar_context/): events.csv, metrics.json, per_class.csv, stitched/*.mp4
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
SOURCE_FPS = 30000 / 1001
OUT_DIR = eg.REPO_ROOT / "outputs" / "gar_context"


def load_clips(clips_dir: Path) -> pd.DataFrame:
    rows = []
    for js in sorted(clips_dir.glob("clip_*.json")):
        mp4 = js.with_suffix(".mp4")
        if not mp4.is_file():
            continue
        j = json.loads(js.read_text())
        c = j["metadata"]["conversion"]
        rows.append({"id": j["id"], "label": j["labels"]["action"]["label"], "game": j["metadata"]["game_id"],
                     "path": str(mp4), "start_frame": c["clip"]["start_frame"],
                     "end_frame": c["clip"]["end_frame_exclusive"],
                     "action_offset_s": c["alignment"]["action_offset_seconds"]})
    if not rows:
        sys.exit(f"No clip_*.mp4 + clip_*.json pairs in {clips_dir}")
    return pd.DataFrame(rows).sort_values(["game", "start_frame"]).reset_index(drop=True)


def find_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Group clips into runs whose source frames overlap or touch (no gaps)."""
    segs, cur, cur_end = [], [], None
    for i, r in df.iterrows():
        if cur and r.game == df.loc[cur[0], "game"] and r.start_frame <= cur_end:
            cur.append(i); cur_end = max(cur_end, r.end_frame)
        else:
            if cur:
                segs.append(df.loc[cur])
            cur, cur_end = [i], r.end_frame
    segs.append(df.loc[cur])
    return segs


def stitch(seg: pd.DataFrame, out: Path) -> tuple[int, int]:
    """Concatenate the segment's clips frame-exactly. Returns (first source frame, n frames)."""
    seg = seg.sort_values("start_frame")
    starts, ends = seg.start_frame.tolist(), seg.end_frame.tolist()
    first, last = starts[0], max(ends)
    parts, pos = [], first                      # pos = next source frame still missing
    for path, s, e in zip(seg.path, starts, ends):
        if e <= pos:
            continue                            # fully covered already
        parts.append((path, pos - s, e - s))    # take [pos, e) from this clip
        pos = e
    if out.is_file():
        return first, last - first
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
    for path, _, _ in parts:
        cmd += ["-i", path]
    chains = [f"[{k}:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{k}]"
              for k, (_, a, b) in enumerate(parts)]
    concat = "".join(f"[v{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    cmd += ["-filter_complex", ";".join(chains + [concat]), "-map", "[out]",
            "-vsync", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return first, last - first


def count_frames(video: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def predict_at(scores: np.ndarray, center_s: float, classes: list[str]) -> tuple[str, float]:
    center = int(round(center_s * tr.SAMPLED_FPS))
    half = int(round(eg.WINDOW_S * tr.SAMPLED_FPS))
    win = scores[max(0, center - half): min(len(scores), center + half + 1), 1:]
    peak = win.max(axis=0) if len(win) else np.zeros(len(tr.CLASSES))
    best = max(classes, key=lambda c: peak[tr.CLASSES.index(c)])
    return best, float(peak[tr.CLASSES.index(best)])


def balanced_acc(df: pd.DataFrame, col: str) -> float:
    classes = [c for c in tr.GAR_CLASSES if (df.label == c).any()]
    return 100 * float(np.mean([(df[df.label == c][col] == c).mean() for c in classes])) if classes else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True, help="folder with clip_*.mp4 + clip_*.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--stitch-only", action="store_true", help="build and check stitched videos, no model")
    a = ap.parse_args()

    df = load_clips(a.clips)
    segs = find_segments(df)
    (OUT_DIR / "stitched").mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} clips from {df.game.nunique()} game(s) -> {len(segs)} continuous segments")

    seg_info = []
    for k, seg in enumerate(segs):
        video = OUT_DIR / "stitched" / f"seg_{seg.game.iloc[0]}_{k:04d}.mp4"
        first, n = stitch(seg, video)
        seg_info.append((seg, video, first, n))
    if a.stitch_only:
        bad = [(v.name, n, count_frames(v)) for _, v, _, n in seg_info if count_frames(v) != n]
        print("frame-count check:", "all segments OK" if not bad else f"MISMATCH {bad[:5]}")
        return

    device = "cuda" if a.device == "auto" and tr.torch.cuda.is_available() else a.device
    model = tr.get_model(device)
    print(f"Model on {tr.pick_device(device)}")
    classes = tr.GAR_CLASSES
    rows, t0 = [], time.perf_counter()
    for k, (seg, video, first, n) in enumerate(seg_info, 1):
        seg_scores = tr.compute_scores(video, model=model, batch_size=1)  # one 8 s window at a time: long segments OOM a T4 with 2
        seg_len_s = n / SOURCE_FPS
        for r in seg.itertuples():
            event_s = (r.start_frame - first) / SOURCE_FPS + r.action_offset_s
            ctx_pred, ctx_conf = predict_at(seg_scores, event_s, classes)
            pad_pred, pad_conf = predict_at(tr.compute_scores(Path(r.path), model=model, batch_size=1), r.action_offset_s, classes)
            rows.append({"id": r.id, "label": r.label, "segment": video.name,
                         "context_before_s": round(event_s, 2), "context_after_s": round(seg_len_s - event_s, 2),
                         "pred_padded": pad_pred, "conf_padded": round(pad_conf, 4),
                         "pred_context": ctx_pred, "conf_context": round(ctx_conf, 4)})
        el = time.perf_counter() - t0
        print(f"  segment {k}/{len(seg_info)} · {len(rows)} events · {el / 60:.1f} min elapsed", flush=True)

    ev = pd.DataFrame(rows)
    ev["min_context_s"] = ev[["context_before_s", "context_after_s"]].min(axis=1)
    ev.to_csv(OUT_DIR / "events.csv", index=False)
    full = ev[ev.min_context_s >= 4.0]          # T-DEED window = 8 s -> 4 s on each side
    metrics = {
        "n_events": len(ev), "n_segments": len(seg_info),
        "balanced_accuracy_padded": balanced_acc(ev, "pred_padded"),
        "balanced_accuracy_context": balanced_acc(ev, "pred_context"),
        "accuracy_padded": 100 * float((ev.label == ev.pred_padded).mean()),
        "accuracy_context": 100 * float((ev.label == ev.pred_context).mean()),
        "n_events_full_context_4s": len(full),
        "balanced_accuracy_padded_full_context_subset": balanced_acc(full, "pred_padded"),
        "balanced_accuracy_context_full_context_subset": balanced_acc(full, "pred_context"),
        "fixed_by_context": int(((ev.pred_padded != ev.label) & (ev.pred_context == ev.label)).sum()),
        "broken_by_context": int(((ev.pred_padded == ev.label) & (ev.pred_context != ev.label)).sum()),
    }
    json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2)
    present = [c for c in tr.GAR_CLASSES if (ev.label == c).any()]
    per = pd.DataFrame({"class": present, "n_events": [int((ev.label == c).sum()) for c in present],
                        "recall_padded": [round((ev[ev.label == c].pred_padded == c).mean(), 3) for c in present],
                        "recall_context": [round((ev[ev.label == c].pred_context == c).mean(), 3) for c in present]})
    per.to_csv(OUT_DIR / "per_class.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(per.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
