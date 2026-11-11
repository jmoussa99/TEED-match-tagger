# Match event tagger

Tags ball actions across a full soccer match with [T-DEED](https://github.com/arturxe2/T-DEED), a ball action spotting model. The app uses the published `SoccerNetBall_challenge2` checkpoint, so it needs no training. It runs locally on Apple Silicon (MPS) or CPU.

The model predicts 12 classes: `PASS`, `DRIVE`, `HEADER`, `HIGH PASS`, `OUT`, `CROSS`, `THROW IN`, `SHOT`, `BALL PLAYER BLOCK`, `PLAYER SUCCESSFUL TACKLE`, `FREE KICK`, `GOAL`. The **GAR 10 classes** filter hides `DRIVE` and `BALL PLAYER BLOCK`, which leaves the GAR challenge label set.

This app is separate from the GAR challenge baseline. The baseline classifies clips that are already centred on an event. T-DEED finds those events in a whole match.

## Setup

Requires the repository `.venv` (see the main README), [uv](https://docs.astral.sh/uv/), and `ffmpeg` (`brew install ffmpeg`). From the repository root:

```bash
bash scripts/setup_tdeed.sh
```

The script does the following:

1. Clones T-DEED into `private/T-DEED` at a pinned commit.
2. Applies [`patches/tdeed_mps.patch`](patches/tdeed_mps.patch). The patch replaces T-DEED's hardcoded CUDA calls so the model runs on MPS or CPU.
3. Installs `gradio` and `gdown`.
4. Downloads the checkpoint (about 185 MB).

T-DEED's own `requirements.txt` is not used, because it pins Linux CUDA wheels.

## Run

UI:

```bash
.venv/bin/python app/ui.py   # open http://127.0.0.1:7860
```

Paste a local path to the match file, or upload it. A path is faster for multi-GB files. Click **Tag match**.

When tagging finishes, you can:

- Filter events by confidence or class without re-running the model.
- Click an event row to play an 8-second preview around it.
- Download the events as JSON or CSV.

Command line:

```bash
.venv/bin/python app/tdeed_runner.py /path/to/match.mp4                 # whole match
.venv/bin/python app/tdeed_runner.py match.mp4 --start 600 --duration 300  # a 5-minute excerpt
```

Results are written to `outputs/events/<video>_events.{json,csv}`. Per-frame scores are cached in `outputs/tdeed_cache/`, so a second run on the same file is instant.

## How it works

1. ffmpeg decodes the video with VideoToolbox. It resamples the video to 12.5 fps and resizes it to 796×448. This matches T-DEED's training frames, which were extracted at 25 fps and sampled every second frame.
2. The model scores frames in 100-frame clips (8 s), with 3/4 overlap between clips. Scores from overlapping clips are averaged.
3. Each class goes through soft non-maximum suppression with a 12-frame window (about 1 s). Each event that remains gets a timestamp from the start of the video and a confidence score.

The windows, padding, displacement-head aggregation and soft-NMS reproduce T-DEED's `inference.py` and `util/eval.py`.

## Notes

- **Accuracy:** T-DEED was trained on SoccerNet broadcast footage (25 fps, 720p). Other camera styles or amateur footage may score lower.
- **Timestamps:** times run from the start of the file. If one file holds both halves, the second half's times include the half-time break.
- **License:** T-DEED is GPL-3.0. It is kept as an external clone under `private/` and is not vendored into this repository.
