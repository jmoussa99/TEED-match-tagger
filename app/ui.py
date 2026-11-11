"""Gradio app: tag ball actions in a full match with T-DEED.

Run from the repository root:
    .venv/bin/python app/ui.py        # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdeed_runner as tr  # noqa: E402

PREVIEW_DIR = tr.CACHE_DIR / "previews"
SERIES_COLOR = "#2a78d6"  # single series; class identity is carried by the axis, not color
PREVIEW_HALF_WINDOW = 4.0  # seconds before/after the event

TABLE_COLUMNS = ["time", "label", "confidence", "seconds"]


def _resolve_video(upload: str | None, local_path: str) -> Path:
    if local_path and local_path.strip():
        p = Path(local_path.strip().strip("'\"")).expanduser()
        if not p.is_file():
            raise gr.Error(f"File not found: {p}")
        return p
    if upload:
        return Path(upload)
    raise gr.Error("Upload a video or paste a local file path.")


def _views(scores, video: str, threshold: float, classes: list[str]):
    """Build table, charts, summary and downloads from cached scores (no model call)."""
    if scores is None:
        empty = pd.DataFrame(columns=TABLE_COLUMNS)
        return empty, None, None, "No match tagged yet.", None, None
    events = tr.extract_events(scores, threshold, classes)
    df = pd.DataFrame(events, columns=["time", "seconds", "position_ms", "label", "confidence"])
    json_path, csv_path = tr.save_events(events, video, threshold)

    counts = pd.DataFrame({
        "label": classes,
        "events": [int((df["label"] == c).sum()) for c in classes],
        "series": "Events",
    }).sort_values("events", ascending=False)
    bar = gr.BarPlot(
        counts, x="label", y="events", color="series", color_map={"Events": SERIES_COLOR},
        colors_in_legend=[], sort="-y", x_title="", y_title="Events", x_label_angle=-35,
        tooltip=["label", "events"], height=320, label="Events per class",
    )

    timeline_df = df.assign(minute=df["seconds"] / 60.0, series="Events")
    timeline = gr.ScatterPlot(
        timeline_df, x="minute", y="label", color="series", color_map={"Events": SERIES_COLOR},
        colors_in_legend=[], x_title="Match time (minutes)", y_title="",
        tooltip=["time", "label", "confidence"], height=360, label="Timeline",
    )

    minutes = len(scores) / tr.SAMPLED_FPS / 60
    summary = (f"**{len(df)} events** in {minutes:.1f} min of video "
               f"(confidence ≥ {threshold:.2f}, {len(classes)} classes shown).")
    return df[TABLE_COLUMNS], bar, timeline, summary, str(json_path), str(csv_path)


def tag_match(upload, local_path, threshold, classes, progress=gr.Progress()):
    video = _resolve_video(upload, local_path)

    def report(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    try:
        scores = tr.compute_scores(video, progress=report)
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e
    except RuntimeError as e:
        raise gr.Error(f"Could not read the video: {e}") from e
    return (scores, str(video), *_views(scores, str(video), threshold, classes))


def refilter(scores, video, threshold, classes):
    return _views(scores, video, threshold, classes)


def preview(video: str | None, evt: gr.SelectData):
    if not video or evt.row_value is None:
        return None
    seconds = float(evt.row_value[TABLE_COLUMNS.index("seconds")])
    start = max(seconds - PREVIEW_HALF_WINDOW, 0.0)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{Path(video).stem}_{int(seconds * 1000)}.mp4"
    if not out.is_file():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", video,
             "-t", f"{2 * PREVIEW_HALF_WINDOW:.2f}", "-vf", "scale=-2:480",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-movflags", "+faststart", str(out)],
            check=True,
        )
    return str(out)


def build() -> gr.Blocks:
    with gr.Blocks(title="Match Event Tagger") as demo:
        gr.Markdown(
            "# Match event tagger\n"
            "Automatic ball-action tagging with T-DEED (SoccerNet Ball Action Spotting). "
            "The first run on a match takes a while; afterwards changing the threshold or classes is instant."
        )
        scores_state = gr.State(None)
        video_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                upload = gr.File(label="Upload match video", file_types=["video"], type="filepath")
                local_path = gr.Textbox(
                    label="…or local file path (faster for large files, no upload copy)",
                    placeholder="/Users/you/Movies/match.mp4",
                )
                run_btn = gr.Button("Tag match", variant="primary")
            with gr.Column(scale=1):
                threshold = gr.Slider(0.05, 0.95, value=tr.DEFAULT_THRESHOLD, step=0.05,
                                      label="Minimum confidence")
                classes = gr.CheckboxGroup(tr.CLASSES, value=tr.CLASSES, label="Classes")
                with gr.Row():
                    all_btn = gr.Button("All 12 classes", size="sm")
                    gar_btn = gr.Button("GAR 10 classes", size="sm")

        summary = gr.Markdown("No match tagged yet.")
        with gr.Row():
            with gr.Column(scale=3):
                table = gr.Dataframe(headers=TABLE_COLUMNS, interactive=False, max_height=520,
                                     label="Events (click a row to preview)", show_search="filter")
            with gr.Column(scale=2):
                clip = gr.Video(label="Event preview", interactive=False, autoplay=True)
                with gr.Row():
                    json_file = gr.File(label="Events JSON", interactive=False)
                    csv_file = gr.File(label="Events CSV", interactive=False)
        with gr.Row():
            bar = gr.BarPlot(label="Events per class")
            timeline = gr.ScatterPlot(label="Timeline")

        view_outputs = [table, bar, timeline, summary, json_file, csv_file]
        run_btn.click(tag_match, [upload, local_path, threshold, classes],
                      [scores_state, video_state, *view_outputs])
        for control in (threshold, classes):
            control.change(refilter, [scores_state, video_state, threshold, classes], view_outputs)
        all_btn.click(lambda: tr.CLASSES, None, classes)
        gar_btn.click(lambda: tr.GAR_CLASSES, None, classes)
        table.select(preview, [video_state], clip)
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", allowed_paths=[str(tr.REPO_ROOT / "outputs")],
    )
