"""Human review of tagged events: confirm, relabel, reject, add missed events, export.

The model's events are recomputed from cached scores whenever the threshold or class filter
changes, so review decisions are kept in a separate store keyed by (model label, time) and
re-applied on every redraw. Events a reviewer adds by hand are kept in the same store and
are always shown, whatever the filters.

Store layout (a plain dict, held in a gr.State):
    {"decisions": {key: {"status": "confirmed" | "corrected" | "rejected", "label": str}},
     "added": [{"seconds": float, "label": str}]}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIRMED, CORRECTED, REJECTED, ADDED = "confirmed", "corrected", "rejected", "added"
MARK = {CONFIRMED: "✓ correct", CORRECTED: "✎ relabeled", REJECTED: "✗ not an event", ADDED: "+ added"}
NO_MODEL_LABEL = "(added by reviewer)"


def new_store() -> dict:
    return {"decisions": {}, "added": []}


def event_key(label: str, seconds: float) -> str:
    return f"{label}@{float(seconds):.2f}"


def decide(store: dict, label: str, seconds: float, status: str, final_label: str | None = None) -> dict:
    store = {"decisions": dict(store["decisions"]), "added": list(store["added"])}
    if label == NO_MODEL_LABEL:                       # a reviewer-added event: rejecting removes it
        if status == REJECTED:
            store["added"] = [a for a in store["added"]
                              if not (abs(a["seconds"] - float(seconds)) < 0.005 and a["label"] == final_label)]
        return store
    store["decisions"][event_key(label, seconds)] = {"status": status, "label": final_label or label}
    return store


def clear(store: dict, label: str, seconds: float) -> dict:
    store = {"decisions": dict(store["decisions"]), "added": list(store["added"])}
    store["decisions"].pop(event_key(label, seconds), None)
    return store


def add(store: dict, seconds: float, label: str) -> dict:
    store = {"decisions": dict(store["decisions"]), "added": list(store["added"])}
    store["added"].append({"seconds": round(float(seconds), 2), "label": label})
    return store


def apply(model_df: pd.DataFrame, store: dict, fmt) -> pd.DataFrame:
    """Model events + review columns + reviewer-added events, sorted by time.

    model_df columns: time, seconds, label, confidence, reliability (others are kept).
    fmt: function seconds -> "mm:ss" used for added events.
    """
    df = model_df.copy()
    dec = [store["decisions"].get(event_key(l, s)) for l, s in zip(df["label"], df["seconds"])]
    df["review"] = [MARK[d["status"]] if d else "" for d in dec]
    df["final label"] = [d["label"] if d and d["status"] != REJECTED else ("" if d else l)
                         for d, l in zip(dec, df["label"])]
    if store["added"]:
        extra = pd.DataFrame([{"time": fmt(a["seconds"]), "seconds": a["seconds"], "label": NO_MODEL_LABEL,
                               "confidence": None, "reliability": "", "review": MARK[ADDED],
                               "final label": a["label"]} for a in store["added"]])
        df = pd.concat([df, extra], ignore_index=True)
    return df.sort_values("seconds", kind="stable").reset_index(drop=True)


def progress_text(df: pd.DataFrame) -> str:
    model_rows = df[df["label"] != NO_MODEL_LABEL]
    done = int((model_rows["review"] != "").sum())
    counts = df["review"].value_counts()
    parts = [f"{counts.get(MARK[s], 0)} {MARK[s].split(' ', 1)[1]}" for s in (CONFIRMED, CORRECTED, REJECTED, ADDED)]
    return f"**Review:** {done} of {len(model_rows)} model events checked · " + ", ".join(parts)


def export(df: pd.DataFrame, video: str | Path, out_dir: Path) -> tuple[Path, Path]:
    """Write the reviewed events. Rejected events are kept with status 'rejected' so the file
    also records the model's mistakes; use rows where status != 'rejected' as the final tags."""
    video = Path(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    status_of = {v: k for k, v in MARK.items()}
    for rec in df.to_dict("records"):          # column names contain spaces, so no itertuples
        status = status_of.get(rec["review"], "unreviewed")
        rows.append({
            "time": rec["time"], "seconds": float(rec["seconds"]),
            "model_label": "" if rec["label"] == NO_MODEL_LABEL else rec["label"],
            "confidence": None if pd.isna(rec["confidence"]) else float(rec["confidence"]),
            "status": status,
            "final_label": "" if status == REJECTED else rec["final label"],
        })
    csv_path = out_dir / f"{video.stem}_reviewed.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({"video": video.name, "events": rows}, indent=2, ensure_ascii=False))
    return csv_path, json_path
