"""How much to trust each T-DEED class on footage it was not trained on.

T-DEED was trained on SoccerNet broadcasts (English league). We measured it on 436 labeled
clips of a different broadcast source (SoccerNet-GAR validation, World Cup 2022 footage from
PFF): see reports/gar_eval_v1/README.md. The recall per class there is the share of real
events of that class that the tagger labels correctly.

Levels (recall): high >= 0.70, medium >= 0.40, low < 0.40. DRIVE and BALL PLAYER BLOCK do not
exist in GAR, so they are untested.

The numbers are read from reports/gar_eval_v1/per_class.csv when it exists (so a newer
evaluation updates the app automatically); otherwise the values below are used.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "reports" / "gar_eval_v1" / "per_class.csv"

# Recall on GAR validation clips (reports/gar_eval_v1, 50 clips per class; 23 goals, 13 high passes).
_FALLBACK_RECALL = {
    "PASS": 0.86, "OUT": 0.76, "SHOT": 0.74, "THROW IN": 0.60, "HEADER": 0.50, "CROSS": 0.30,
    "PLAYER SUCCESSFUL TACKLE": 0.0, "FREE KICK": 0.0, "GOAL": 0.0, "HIGH PASS": 0.0,
}

HIGH, MEDIUM, LOW, UNTESTED = "high", "medium", "low", "untested"
_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2, UNTESTED: 3}


def _load_recall() -> dict[str, float]:
    if RESULTS_CSV.is_file():
        with RESULTS_CSV.open() as f:
            rows = {r["class"]: float(r["recall"]) for r in csv.DictReader(f) if r.get("recall")}
        if rows:
            return rows
    return dict(_FALLBACK_RECALL)


RECALL = _load_recall()


def level(label: str) -> str:
    r = RECALL.get(label)
    if r is None:
        return UNTESTED
    return HIGH if r >= 0.70 else MEDIUM if r >= 0.40 else LOW


def classes_at_least(classes: list[str], min_level: str = MEDIUM) -> list[str]:
    """Classes (kept in the given order) whose reliability is min_level or better."""
    return [c for c in classes if _ORDER[level(c)] <= _ORDER[min_level]]


def legend_markdown(classes: list[str]) -> str:
    rows = sorted(classes, key=lambda c: (_ORDER[level(c)], -(RECALL.get(c) or 0)))
    lines = [
        "Measured on labeled clips from a **different broadcast source** than T-DEED was trained on "
        "(World Cup 2022 footage, SoccerNet-GAR). *Recall* = share of real events of that class the tagger "
        "labels correctly. Treat **low** classes as unreliable: most real tackles, free kicks, goals and "
        "high passes are missed or labeled as something else (goals usually appear as SHOT).",
        "",
        "| Class | Reliability | Recall |",
        "|---|---|---|",
    ]
    for c in rows:
        r = RECALL.get(c)
        lines.append(f"| {c} | {level(c)} | {'—' if r is None else f'{r:.0%}'} |")
    return "\n".join(lines)
