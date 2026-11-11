"""Tag ball actions in a full soccer match with T-DEED (SoccerNet Ball Action Spotting).

Runs the published ``SoccerNetBall_challenge2`` checkpoint from a patched T-DEED clone in
``private/T-DEED`` (see ``scripts/setup_tdeed.sh``). Frames are streamed from ffmpeg, so the
match is never transcoded to disk. Per-frame class scores are cached, so changing the
confidence threshold only re-runs the cheap post-processing.

The pipeline mirrors T-DEED's ``inference.py`` and ``util/eval.py::inference``:
796x448 RGB frames at 12.5 fps (25 fps source, stride 2), 100-frame clips with 3/4 overlap,
5 zero frames of leading padding, displacement-head aggregation, score averaging across
overlapping clips, then per-class soft non-maximum suppression with a 12-frame window.

CLI:
    .venv/bin/python app/tdeed_runner.py match.mp4 [--start 0] [--duration 300] [--threshold 0.2]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Callable, Iterator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TDEED_DIR = REPO_ROOT / "private" / "T-DEED"
MODEL_NAME = "SoccerNetBall_challenge2"
CACHE_DIR = REPO_ROOT / "outputs" / "tdeed_cache"
EVENTS_DIR = REPO_ROOT / "outputs" / "events"

SOURCE_FPS = 25.0                    # training frames were extracted at 25 fps
STRIDE = 2                           # STRIDE_SNB in T-DEED: every 2nd frame
SAMPLED_FPS = SOURCE_FPS / STRIDE    # 12.5 frames per second reach the model
FRAME_W, FRAME_H = 796, 448          # training resolution (extract_frames_snb.py)
CLIP_LEN = 100                       # from the challenge config
CLIP_STEP = CLIP_LEN - CLIP_LEN // 4 * 3   # inference.py overlaps clips by 3/4 -> step 25
PAD_LEN = 5                          # DEFAULT_PAD_LEN in dataset/frame.py
NMS_WINDOW = 12                      # WINDOWS_SNB[1], in sampled frames
DEFAULT_THRESHOLD = 0.2              # inference.py default

CLASSES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
    "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL",
]  # data/soccernetball/class.txt; index i in CLASSES is score column i + 1 (0 = background)
GAR_CLASSES = [c for c in CLASSES if c not in ("DRIVE", "BALL PLAYER BLOCK")]

ProgressFn = Callable[[float, str], None]


def _no_progress(_fraction: float, _message: str) -> None:
    pass


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- model

def load_model(device: torch.device | str = "auto"):
    """Build T-DEED from the challenge config and load the checkpoint (tensors only)."""
    device = pick_device(device) if isinstance(device, str) else device
    ckpt = TDEED_DIR / "checkpoints" / "SoccerNetBall" / MODEL_NAME / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} missing; run scripts/setup_tdeed.sh")

    if str(TDEED_DIR) not in sys.path:
        sys.path.insert(0, str(TDEED_DIR))
    from model.model import TDEEDModel  # noqa: PLC0415  (T-DEED package)

    config = json.loads((TDEED_DIR / "config" / "SoccerNetBall" / f"{MODEL_NAME}.json").read_text())
    args = types.SimpleNamespace(**config)
    args.crop_dim = None if config["crop_dim"] <= 0 else config["crop_dim"]
    args.pretrained_backbone = False  # weights come from the checkpoint; skip the ImageNet download
    if args.num_classes != len(CLASSES):
        raise ValueError(f"config has {args.num_classes} classes, expected {len(CLASSES)}")

    model = TDEEDModel(device=device, args=args)
    if config.get("pretrain"):
        # Same as inference.py: second head for the SoccerNet pretraining classes.
        model._model.update_pred_head([len(CLASSES) + 1, config["pretrain"]["num_classes"] + 1])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model._model.load_state_dict(state, strict=True)
    model._model.eval()
    return model


_MODELS: dict[str, object] = {}


def get_model(device: str = "auto"):
    """Load the model once per device and reuse it (the UI calls this on every run)."""
    key = str(pick_device(device))
    if key not in _MODELS:
        _MODELS[key] = load_model(key)
    return _MODELS[key]


def _displaced_max(probs: torch.Tensor, displ: torch.Tensor) -> torch.Tensor:
    """Vectorised ``model.modules.process_double_head`` (moves each frame's scores by its
    predicted displacement and keeps the per-frame maximum)."""
    b, t, c = probs.shape
    target = (torch.arange(t).expand(b, t) - displ.round().long()).clamp(0, t - 1)
    out = torch.zeros_like(probs)
    out.scatter_reduce_(1, target.unsqueeze(-1).expand(b, t, c), probs, reduce="amax", include_self=True)
    return out


@torch.no_grad()
def _predict_clips(model, clips: np.ndarray) -> np.ndarray:
    """clips: uint8 (B, T, H, W, 3) -> per-frame scores (B, T, 1 + num_classes)."""
    x = torch.from_numpy(clips).to(model.device).permute(0, 1, 4, 2, 3).float()
    out, _ = model._model(x, inference=True)
    logits = out["im_feat"][..., : len(CLASSES) + 1].float().cpu()
    displ = out["displ_feat"].float().cpu()
    return _displaced_max(torch.softmax(logits, dim=2), displ).numpy()


# --------------------------------------------------------------------------- video

def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out and out != "N/A" else 0.0


def iter_frames(video: Path, start: float = 0.0, duration: float | None = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (H, W, 3) resampled to 12.5 fps at 796x448."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-hwaccel", "videotoolbox"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-an", "-sn", "-dn",
        "-vf", f"fps={SAMPLED_FPS},scale={FRAME_W}:{FRAME_H}:flags=bilinear",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    frame_bytes = FRAME_W * FRAME_H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    finished = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                finished = True
                break
            yield np.frombuffer(buf, np.uint8).reshape(FRAME_H, FRAME_W, 3)
    finally:
        if not finished and proc.poll() is None:
            proc.terminate()  # consumer stopped early
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {err.strip()[-800:]}")


def iter_clips(frames: Iterator[np.ndarray]) -> Iterator[tuple[np.ndarray, int]]:
    """Same windows as dataset/frame.py::ActionSpotInferenceDataset, plus one final
    zero-padded clip so the last seconds of the match are scored too."""
    blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    buffer: list[np.ndarray] = [blank] * PAD_LEN
    n_real = 0          # real frames appended so far
    covered_until = 0   # real frames covered by an emitted clip
    for frame in frames:
        buffer.append(frame)
        n_real += 1
        if len(buffer) == CLIP_LEN:
            yield np.stack(buffer), n_real - CLIP_LEN
            covered_until = n_real
            buffer = buffer[CLIP_STEP:]
    if n_real > covered_until:
        start = n_real - len(buffer)
        buffer += [blank] * (CLIP_LEN - len(buffer))
        yield np.stack(buffer), start


# --------------------------------------------------------------------------- scoring

def _cache_key(video: Path, start: float, duration: float | None) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}|{start}|{duration}|{MODEL_NAME}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def compute_scores(
    video: Path | str,
    model=None,
    start: float = 0.0,
    duration: float | None = None,
    batch_size: int = 2,
    device: str = "auto",
    progress: ProgressFn = _no_progress,
    use_cache: bool = True,
) -> np.ndarray:
    """Return averaged per-frame scores (N, 1 + 12) at 12.5 fps; row k is time start + k/12.5 s."""
    video = Path(video)
    cache = CACHE_DIR / f"{_cache_key(video, start, duration)}.npy"
    if use_cache and cache.is_file():
        progress(1.0, "Loaded cached scores")
        return np.load(cache)

    if model is None:
        progress(0.0, "Loading model")
        model = get_model(device)

    total = duration or max(probe_duration(video) - start, 0.0)
    est_frames = int(total * SAMPLED_FPS) + CLIP_LEN
    scores = np.zeros((est_frames, len(CLASSES) + 1), np.float32)
    support = np.zeros(est_frames, np.int32)
    n_frames = 0

    def flush(batch: list[np.ndarray], starts: list[int]) -> None:
        nonlocal scores, support
        preds = _predict_clips(model, np.stack(batch))
        for pred, s in zip(preds, starts):
            if s < 0:
                pred, s = pred[-s:], 0
            end = s + len(pred)
            if end > len(scores):  # duration estimate was short; grow
                grow = end - len(scores) + 10 * CLIP_LEN
                scores = np.concatenate([scores, np.zeros((grow, scores.shape[1]), np.float32)])
                support = np.concatenate([support, np.zeros(grow, np.int32)])
            scores[s:end] += pred
            support[s:end] += (pred.sum(axis=1) != 0).astype(np.int32)

    def counting(frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        nonlocal n_frames
        for f in frames:
            n_frames += 1
            yield f

    t0 = time.perf_counter()
    batch: list[np.ndarray] = []
    starts: list[int] = []
    for clip, s in iter_clips(counting(iter_frames(video, start, duration))):
        batch.append(clip)
        starts.append(s)
        if len(batch) == batch_size:
            flush(batch, starts)
            batch, starts = [], []
            done = n_frames / SAMPLED_FPS
            rate = done / max(time.perf_counter() - t0, 1e-6)
            eta = (total - done) / rate if rate > 0 else 0
            progress(min(done / total, 0.999) if total else 0.0,
                     f"{_fmt(done)} / {_fmt(total)} processed · {rate:.1f}x real time · ETA {_fmt(eta)}")
    if batch:
        flush(batch, starts)

    scores, support = scores[:n_frames], support[:n_frames]
    support[support == 0] = 1
    scores /= support[:, None]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    progress(1.0, f"Done in {_fmt(time.perf_counter() - t0)}")
    return scores


def _soft_nms_1d(score: np.ndarray, threshold: float, window: int) -> list[tuple[int, float]]:
    """Dense equivalent of util/eval.py::soft_non_maximum_supression for one class."""
    s = np.where(score >= threshold, score, -1.0).astype(np.float64)
    picked = []
    while True:
        i = int(np.argmax(s))
        if s[i] < threshold:
            break
        picked.append((i, float(s[i])))
        lo, hi = max(0, i - window), min(len(s), i + window + 1)
        seg = s[lo:hi]
        dist = np.abs(np.arange(lo, hi) - i)
        live = seg >= 0
        seg[live] *= dist[live] ** 2 / window ** 2
        s[i] = -1.0
    return picked


def extract_events(
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    classes: list[str] | None = None,
    start: float = 0.0,
) -> list[dict]:
    """Per-class soft-NMS over the score track; returns events sorted by time."""
    wanted = set(classes or CLASSES)
    events = []
    for j, label in enumerate(CLASSES, start=1):
        if label not in wanted:
            continue
        for frame, conf in _soft_nms_1d(scores[:, j], threshold, NMS_WINDOW):
            t = start + frame / SAMPLED_FPS
            events.append({"time": _fmt(t), "seconds": round(t, 2), "position_ms": int(round(t * 1000)),
                           "label": label, "confidence": round(conf, 4)})
    events.sort(key=lambda e: (e["seconds"], -e["confidence"]))
    return events


def save_events(events: list[dict], video: Path | str, threshold: float, out: Path | None = None) -> tuple[Path, Path]:
    video = Path(video)
    out = out or EVENTS_DIR / f"{video.stem}_events.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"video": video.name, "model": f"T-DEED {MODEL_NAME}", "threshold": threshold,
           "sampled_fps": SAMPLED_FPS, "num_events": len(events), "predictions": events}
    out.write_text(json.dumps(doc, indent=2))
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["time", "seconds", "position_ms", "label", "confidence"])
        w.writeheader()
        w.writerows(events)
    return out, csv_path


def _fmt(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    p.add_argument("--duration", type=float, default=None, help="only process this many seconds")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="events JSON path (CSV written alongside)")
    a = p.parse_args()

    last = [0.0]

    def report(frac: float, msg: str) -> None:
        if time.perf_counter() - last[0] > 5 or frac >= 1.0:
            print(f"[{frac:6.1%}] {msg}", flush=True)
            last[0] = time.perf_counter()

    t0 = time.perf_counter()
    scores = compute_scores(a.video, start=a.start, duration=a.duration, batch_size=a.batch_size,
                            device=a.device, progress=report, use_cache=not a.no_cache)
    events = extract_events(scores, a.threshold, start=a.start)
    json_path, csv_path = save_events(events, a.video, a.threshold, a.out)
    covered = len(scores) / SAMPLED_FPS
    wall = time.perf_counter() - t0
    print(f"{len(events)} events over {_fmt(covered)} of video in {_fmt(wall)} "
          f"({covered / max(wall, 1e-6):.1f}x real time)")
    counts = {c: sum(e["label"] == c for e in events) for c in CLASSES}
    print("  " + ", ".join(f"{c}: {n}" for c, n in counts.items() if n))
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
