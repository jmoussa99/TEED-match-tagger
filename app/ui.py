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
import reliability as rel  # noqa: E402
import review as rv  # noqa: E402
import tdeed_runner as tr  # noqa: E402

PREVIEW_DIR = tr.CACHE_DIR / "previews"
REVIEW_DIR = tr.REPO_ROOT / "outputs" / "events"
SERIES_COLOR = "#2a78d6"  # single series; class identity is carried by the axis, not color
FULL_VIDEO_HEIGHT = 480   # the full-match player streams a light copy, not the original file

TABLE_COLUMNS = ["time", "label", "confidence", "reliability", "review", "final label", "seconds"]

# Browser-side helpers for the full-match player (elem_id="full-video").
JS_SEEK = """(t) => { const v = document.querySelector('#full-video video');
  if (v && t !== null && t !== undefined) { v.currentTime = Math.max(Number(t) - 2, 0); } }"""
JS_CURRENT_TIME = """(t) => { const v = document.querySelector('#full-video video');
  return v ? Math.round(v.currentTime * 100) / 100 : t; }"""


def _resolve_video(upload: str | None, local_path: str) -> Path:
    if local_path and local_path.strip():
        p = Path(local_path.strip().strip("'\"")).expanduser()
        if not p.is_file():
            raise gr.Error(f"File not found: {p}")
        return p
    if upload:
        return Path(upload)
    raise gr.Error("Upload a video or paste a local file path.")


def _views(scores, video: str, threshold: float, classes: list[str], store: dict | None = None):
    """Build table, charts, summary and downloads from cached scores (no model call)."""
    if scores is None:
        empty = pd.DataFrame(columns=TABLE_COLUMNS)
        return empty, None, None, "No match tagged yet.", None, None
    events = tr.extract_events(scores, threshold, classes)
    df = pd.DataFrame(events, columns=["time", "seconds", "position_ms", "label", "confidence"])
    df["reliability"] = df["label"].map(rel.level)
    json_path, csv_path = tr.save_events(events, video, threshold)
    table = rv.apply(df, store or rv.new_store(), tr._fmt)

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
    shaky = df[df["reliability"].isin([rel.LOW, rel.UNTESTED])]
    if len(shaky):
        names = ", ".join(sorted(shaky["label"].unique()))
        summary += (f"\n\n⚠️ {len(shaky)} of these events are from classes with **low or untested reliability** "
                    f"({names}). Check them in the preview before relying on them.")
    summary += "\n\n" + rv.progress_text(table)
    return table[TABLE_COLUMNS], bar, timeline, summary, str(json_path), str(csv_path)


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
    store = rv.new_store()
    return (scores, str(video), store, None, *_views(scores, str(video), threshold, classes, store))


def refilter(scores, video, threshold, classes, store):
    return _views(scores, video, threshold, classes, store)


def load_full_video(video: str | None, progress=gr.Progress()):
    """Make a light 480p copy of the whole match for the in-browser player (streams fast over a
    share link, and works for any input format). Takes a while for a full match; cached after."""
    if not video:
        raise gr.Error("Tag a match first.")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{Path(video).stem}_full{FULL_VIDEO_HEIGHT}p.mp4"
    if not out.is_file():
        progress(0.1, desc="Making a 480p copy of the full video…")
        tmp = out.with_suffix(".part.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"scale=-2:{FULL_VIDEO_HEIGHT}",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)],
            check=True,
        )
        tmp.rename(out)
    return str(out)


def select_event(evt: gr.SelectData):
    """Remember the clicked row for the review buttons and return its time for the player."""
    if evt.row_value is None:
        return None, "Click an event in the table.", None, None
    row = dict(zip(TABLE_COLUMNS, evt.row_value))
    sel = {"label": row["label"], "seconds": float(row["seconds"]), "time": row["time"],
           "final": row["final label"] or row["label"]}
    who = "added by you" if row["label"] == rv.NO_MODEL_LABEL else f"model says **{row['label']}**"
    note = f"**{row['time']}** · {who}" + (f" · {row['review']}" if row["review"] else "")
    default = sel["final"] if sel["final"] in tr.CLASSES else None
    return sel, note, gr.update(value=default), sel["seconds"]


def _review(store, sel, scores, video, threshold, classes, status, new_label=None):
    if not sel:
        raise gr.Error("Click an event in the table first.")
    if status == rv.CORRECTED and not new_label:
        raise gr.Error("Pick a class under 'Relabel as' first.")
    if status == "undo":
        store = rv.clear(store, sel["label"], sel["seconds"])
        done = "review cleared"
    else:
        store = rv.decide(store, sel["label"], sel["seconds"], status,
                          new_label if status == rv.CORRECTED else sel["final"])
        done = rv.MARK[status] + (f" → {new_label}" if status == rv.CORRECTED else "")
    note = f"**{sel['time']}** · {done}. Click the next event."
    return (store, *_views(scores, video, threshold, classes, store), note)


def add_event(store, seconds, label, scores, video, threshold, classes):
    if scores is None:
        raise gr.Error("Tag a match first.")
    if seconds is None or label is None:
        raise gr.Error("Set a time and a class for the missed event.")
    store = rv.add(store, seconds, label)
    note = f"**{tr._fmt(seconds)}** · {rv.MARK[rv.ADDED]} {label}."
    return (store, *_views(scores, video, threshold, classes, store), note)


def export_review(scores, video, threshold, classes, store):
    if scores is None:
        raise gr.Error("Tag a match first.")
    table = _views(scores, video, threshold, classes, store)[0]
    csv_path, json_path = rv.export(table, video, REVIEW_DIR)
    return gr.update(value=[str(csv_path), str(json_path)], visible=True)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Linux / Colab
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",          # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
EVENT_FLASH = 0.3  # seconds before/after the event that the red frame is shown


def _font() -> str:
    for f in FONT_CANDIDATES:
        if Path(f).is_file():
            return f":fontfile='{f}'"
    return ""  # let ffmpeg/fontconfig pick a default


def _esc(text: str) -> str:
    """Escape text for ffmpeg drawtext."""
    return text.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:").replace("%", "\\%")


def preview(video: str | None, length: float, slow: bool, evt: gr.SelectData):
    if not video or evt.row_value is None:
        return None
    row = dict(zip(TABLE_COLUMNS, evt.row_value))
    return _make_preview(video, float(row["seconds"]), row["time"], row["final label"] or row["label"], length, slow)


def repreview(video: str | None, length: float, slow: bool, sel: dict | None):
    """Re-cut the selected event's clip when the preview length or slow motion changes."""
    if not video or not sel:
        return gr.update()
    return _make_preview(video, sel["seconds"], sel["time"], sel["final"], length, slow)


def _make_preview(video: str, seconds: float, time_text: str, label: str, length: float, slow: bool) -> str:
    """Cut a short clip around an event and mark the event moment: the label and time stay in the
    corner, and a red frame + 'EVENT' flash for 0.6 s at the tagged time."""
    length = float(length)
    half = length / 2
    start = max(seconds - half, 0.0)
    at = seconds - start                          # event time inside the clip
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch for ch in label if ch.isalnum())[:12]
    out = PREVIEW_DIR / f"{Path(video).stem}_{int(seconds * 1000)}_{int(length)}s{'_slow' if slow else ''}_{tag}.mp4"
    if out.is_file():
        return str(out)
    on = f"enable='between(t,{max(at - EVENT_FLASH, 0):.2f},{at + EVENT_FLASH:.2f})'"
    font = _font()
    marks = [
        f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=10:{on}",
        f"drawtext=text='EVENT':x=(w-text_w)/2:y=24:fontsize=40:fontcolor=white:box=1:boxcolor=red@0.85:boxborderw=10{font}:{on}",
        f"drawtext=text='{_esc(label)} @ {_esc(time_text)}':x=16:y=h-text_h-16:fontsize=24:fontcolor=white"
        f":box=1:boxcolor=black@0.6:boxborderw=8{font}",
    ]
    base = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", video]  # -t on the input, so slow motion is not cut short
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    speed = ",setpts=2.0*PTS" if slow else ""
    audio = ["-an"] if slow else ["-c:a", "aac"]
    try:
        subprocess.run(base + ["-vf", "scale=-2:480," + ",".join(marks) + speed] + enc + audio + [str(out)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # ffmpeg without drawtext/freetype: keep the red frame, drop the text
        subprocess.run(base + ["-vf", "scale=-2:480," + marks[0] + speed] + enc + audio + [str(out)], check=True)
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
        review_state = gr.State(rv.new_store())
        selected_state = gr.State(None)

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
                    reliable_btn = gr.Button("Reliable classes only", size="sm")
                with gr.Accordion("How reliable is each class?", open=False):
                    gr.Markdown(rel.legend_markdown(tr.CLASSES))

        summary = gr.Markdown("No match tagged yet.")
        with gr.Row():
            with gr.Column(scale=3):
                table = gr.Dataframe(headers=TABLE_COLUMNS, interactive=False, max_height=520,
                                     label="Events (click a row to preview)", show_search="filter")
            with gr.Column(scale=2):
                clip = gr.Video(label="Event preview (red frame = the tagged moment)",
                                interactive=False, autoplay=True, height=280)
                with gr.Row(equal_height=True):
                    preview_len = gr.Radio([4, 6, 8], value=4, label="Preview length (s)", scale=3, min_width=0)
                    preview_slow = gr.Checkbox(value=False, label="Slow motion (½×)", scale=2, min_width=0)
                selected_note = gr.Markdown("Click an event in the table to preview and review it.")
                sel_seconds = gr.Number(visible=False)
                with gr.Group():
                    with gr.Row(equal_height=True):
                        confirm_btn = gr.Button("✓ Correct", size="sm", min_width=0)
                        reject_btn = gr.Button("✗ Not an event", size="sm", min_width=0)
                        undo_btn = gr.Button("↺ Undo", size="sm", min_width=0)
                    with gr.Row(equal_height=True):
                        fix_label = gr.Dropdown(tr.CLASSES, value=None, label="Relabel as", scale=3, min_width=0)
                        relabel_btn = gr.Button("✎ Relabel", size="sm", scale=1, min_width=0)
                gr.Markdown("**Missed an event?** Pause the full video at that moment, then add it:")
                with gr.Group():
                    with gr.Row(equal_height=True):
                        add_time = gr.Number(label="Time (s)", precision=2, scale=3, min_width=0)
                        now_btn = gr.Button("⏱ Use video time", size="sm", scale=1, min_width=0)
                    with gr.Row(equal_height=True):
                        add_label = gr.Dropdown(tr.CLASSES, value=None, label="Class", scale=3, min_width=0)
                        add_btn = gr.Button("+ Add", size="sm", scale=1, min_width=0)
                export_btn = gr.Button("Export reviewed events")
                reviewed_files = gr.File(label="Reviewed events", file_count="multiple",
                                         interactive=False, visible=False)

        with gr.Accordion("Full video", open=False):
            gr.Markdown("Loads a 480p copy of the whole match (a few minutes for a full match, then cached). "
                        "Clicking an event jumps the player to 2 s before it.")
            full_btn = gr.Button("Load full video", size="sm")
            full_video = gr.Video(label="Full match", interactive=False, elem_id="full-video", height=420)

        with gr.Accordion("Model output files (JSON / CSV)", open=False):
            with gr.Row():
                json_file = gr.File(label="Events JSON", interactive=False)
                csv_file = gr.File(label="Events CSV", interactive=False)

        with gr.Row():
            bar = gr.BarPlot(label="Events per class")
            timeline = gr.ScatterPlot(label="Timeline")

        view_outputs = [table, bar, timeline, summary, json_file, csv_file]
        filters = [scores_state, video_state, threshold, classes]
        run_btn.click(tag_match, [upload, local_path, threshold, classes],
                      [scores_state, video_state, review_state, full_video, *view_outputs])
        for control in (threshold, classes):
            control.change(refilter, [*filters, review_state], view_outputs)
        all_btn.click(lambda: tr.CLASSES, None, classes)
        gar_btn.click(lambda: tr.GAR_CLASSES, None, classes)
        reliable_btn.click(lambda: rel.classes_at_least(tr.CLASSES, rel.MEDIUM), None, classes)
        table.select(preview, [video_state, preview_len, preview_slow], clip)
        for control in (preview_len, preview_slow):
            control.change(repreview, [video_state, preview_len, preview_slow, selected_state], clip)
        table.select(select_event, None, [selected_state, selected_note, fix_label, sel_seconds]).then(
            None, [sel_seconds], None, js=JS_SEEK)

        review_inputs = [review_state, selected_state, *filters]
        review_outputs = [review_state, *view_outputs, selected_note]
        confirm_btn.click(lambda *a: _review(*a, rv.CONFIRMED), review_inputs, review_outputs)
        relabel_btn.click(lambda *a: _review(*a[:-1], rv.CORRECTED, a[-1]), [*review_inputs, fix_label], review_outputs)
        reject_btn.click(lambda *a: _review(*a, rv.REJECTED), review_inputs, review_outputs)
        undo_btn.click(lambda *a: _review(*a, "undo"), review_inputs, review_outputs)
        now_btn.click(None, [add_time], [add_time], js=JS_CURRENT_TIME)
        add_btn.click(add_event, [review_state, add_time, add_label, *filters], review_outputs)
        export_btn.click(export_review, [*filters, review_state], reviewed_files)
        full_btn.click(load_full_video, [video_state], full_video)
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", allowed_paths=[str(tr.REPO_ROOT / "outputs")],
    )
