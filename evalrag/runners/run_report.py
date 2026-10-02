"""
Compare plusieurs runs d'évaluation sur plusieurs scores.

Version allégée du rapport réel : même logique d'appariement et d'agrégation,
mais sortie Markdown au lieu d'un classeur Excel mis en forme.

Exemple :
    python -m evalrag.runners.run_report --dataset_name golden_set_demo \
        --runs <run_A> <run_B> --score_names faithfulness context_recall
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evalrag.config import PACKAGE_ROOT, load_config
from evalrag.stubs.langfuse_client import get_langfuse_client


@dataclass(frozen=True)
class RunItemResult:
    dataset_item_id: str
    trace_id: str
    run_name: str
    label: str
    scores_by_name: dict[str, float | None]


@dataclass(frozen=True)
class ItemComparison:
    dataset_item_id: str
    label: str
    scores_by_run: dict[str, dict[str, float | None]]


@dataclass(frozen=True)
class RunScoreSummary:
    run_name: str
    score_name: str
    item_count: int
    available_count: int
    missing_count: int
    mean_score: float | None
    min_score: float | None
    max_score: float | None


@dataclass(frozen=True)
class ScoreItemRow:
    dataset_item_id: str
    label: str
    score_name: str
    scores_by_run: dict[str, float | None]
    available_count: int
    mean_score: float | None
    min_score: float | None
    max_score: float | None
    score_range: float | None
    worst_runs: list[str]
    best_runs: list[str]


def _coerce_score_value(value: object | None) -> float | None:
    """
    Convertit une valeur de score en float si possible.

    Bool est converti en 0.0 / 1.0.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _truncate_text(value: str, max_length: int = 80) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def _extract_requested_scores(trace: Mapping, requested_score_names: Sequence[str], trace_id: str) -> dict[str, float | None]:
    """
    Extrait exactement les scores demandés depuis une trace.

    - si un score demandé n'existe pas : valeur None
    - si plusieurs scores du même nom existent sur la même trace : erreur
    """
    raw_scores = trace.get("scores") or []
    result: dict[str, float | None] = {name: None for name in requested_score_names}

    for score_name in requested_score_names:
        matching_values: list[float] = []
        for raw_score in raw_scores:
            if str(raw_score.get("name")) != score_name:
                continue
            value = _coerce_score_value(raw_score.get("value"))
            if value is not None:
                matching_values.append(value)

        if len(matching_values) > 1:
            raise ValueError(
                f"Trace '{trace_id}': plusieurs valeurs trouvées pour le score "
                f"'{score_name}'. La comparaison serait ambiguë."
            )
        if len(matching_values) == 1:
            result[score_name] = matching_values[0]

    return result


def _load_run_results(langfuse_client, dataset_name: str, run_name: str, score_names: Sequence[str]) -> dict[str, RunItemResult]:
    """Charge un run complet et retourne ses items indexés par dataset_item_id."""
    run_record = langfuse_client.get_run(dataset_name=dataset_name, run_name=run_name)
    indexed_results: dict[str, RunItemResult] = {}

    for raw_run_item in run_record.get("dataset_run_items", []):
        dataset_item_id = raw_run_item["dataset_item_id"]
        trace_id = raw_run_item["trace_id"]

        if dataset_item_id in indexed_results:
            raise ValueError(f"Run '{run_name}': dataset_item_id dupliqué détecté: '{dataset_item_id}'.")

        trace = langfuse_client.get_trace(trace_id)
        label = _truncate_text(str(trace.get("input") or dataset_item_id))
        indexed_results[dataset_item_id] = RunItemResult(
            dataset_item_id=dataset_item_id,
            trace_id=trace_id,
            run_name=run_name,
            label=label,
            scores_by_name=_extract_requested_scores(trace, score_names, trace_id),
        )

    return indexed_results


def _get_common_item_ids(results_by_run: dict[str, dict[str, RunItemResult]]) -> list[str]:
    """Retourne les dataset_item_id présents dans tous les runs."""
    run_names = list(results_by_run.keys())
    if not run_names:
        return []
    common_ids = set(results_by_run[run_names[0]].keys())
    for run_name in run_names[1:]:
        common_ids &= set(results_by_run[run_name].keys())
    return sorted(common_ids)


def _build_item_comparisons(runs, common_item_ids, results_by_run) -> list[ItemComparison]:
    comparisons: list[ItemComparison] = []
    for dataset_item_id in common_item_ids:
        first_item = results_by_run[runs[0]][dataset_item_id]
        scores_by_run = {run: dict(results_by_run[run][dataset_item_id].scores_by_name) for run in runs}
        comparisons.append(ItemComparison(dataset_item_id, first_item.label, scores_by_run))
    return comparisons


def _compute_run_score_summaries(comparisons, runs, score_name) -> list[RunScoreSummary]:
    """Calcule un résumé par run pour un score donné."""
    summaries: list[RunScoreSummary] = []

    for run_name in runs:
        values: list[float] = []
        for comparison in comparisons:
            value = comparison.scores_by_run[run_name][score_name]
            if value is not None:
                values.append(value)

        item_count = len(comparisons)
        available_count = len(values)
        mean_score = min_score = max_score = None
        if values:
            mean_score = sum(values) / len(values)
            min_score = min(values)
            max_score = max(values)

        summaries.append(RunScoreSummary(
            run_name, score_name, item_count, available_count, item_count - available_count,
            mean_score, min_score, max_score,
        ))

    return summaries


def _build_score_item_rows(comparisons, runs, score_name) -> list[ScoreItemRow]:
    rows: list[ScoreItemRow] = []
    for comparison in comparisons:
        scores_by_run = {run: comparison.scores_by_run[run][score_name] for run in runs}
        available = {run: s for run, s in scores_by_run.items() if s is not None}

        if available:
            values = list(available.values())
            min_score, max_score = min(values), max(values)
            mean_score = sum(values) / len(values)
            score_range = max_score - min_score
            worst_runs = [r for r, s in available.items() if s == min_score]
            best_runs = [r for r, s in available.items() if s == max_score]
        else:
            min_score = max_score = mean_score = score_range = None
            worst_runs, best_runs = [], []

        rows.append(ScoreItemRow(
            comparison.dataset_item_id, comparison.label, score_name, scores_by_run, len(available),
            mean_score, min_score, max_score, score_range, worst_runs, best_runs,
        ))
    return rows


def _get_worst_rows(rows: Sequence[ScoreItemRow]) -> list[ScoreItemRow]:
    """Items les moins bons globalement (moyenne croissante, puis minimum croissant)."""
    return sorted(rows, key=lambda row: (
        row.mean_score is None,
        float("inf") if row.mean_score is None else row.mean_score,
        float("inf") if row.min_score is None else row.min_score,
        row.label.lower(),
        row.dataset_item_id,
    ))


def _get_diff_rows(rows: Sequence[ScoreItemRow]) -> list[ScoreItemRow]:
    """Items pour lesquels les runs diffèrent (écart max - min strictement positif)."""
    filtered = [r for r in rows if r.available_count >= 2 and r.score_range is not None and r.score_range > 0.0]
    return sorted(filtered, key=lambda row: (
        -(row.score_range if row.score_range is not None else -1.0),
        float("inf") if row.mean_score is None else row.mean_score,
        row.label.lower(),
        row.dataset_item_id,
    ))


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _render_markdown(dataset_name, runs, score_names, top_n, analysis) -> str:
    lines = [f"# Comparaison de runs : {dataset_name}", "", f"Runs : {', '.join(runs)}", ""]
    for score_name in score_names:
        a = analysis[score_name]
        lines += [f"## {score_name}", "", "| run | n | dispo | manquants | moyenne | min | max |",
                  "|---|---|---|---|---|---|---|"]
        for s in a["summaries"]:
            lines.append(f"| {s.run_name} | {s.item_count} | {s.available_count} | {s.missing_count} | "
                         f"{_fmt(s.mean_score)} | {_fmt(s.min_score)} | {_fmt(s.max_score)} |")
        lines += ["", f"Pires items (top {top_n}) :", ""]
        for r in a["worst_rows"][:top_n]:
            lines.append(f"- {r.label} : moyenne {_fmt(r.mean_score)} | " +
                         " | ".join(f"{k}={_fmt(v)}" for k, v in r.scores_by_run.items()))
        lines += ["", f"Items qui diffèrent entre runs (top {top_n}) :", ""]
        for r in a["diff_rows"][:top_n]:
            lines.append(f"- {r.label} : écart {_fmt(r.score_range)} | pire={','.join(r.worst_runs)}")
        lines.append("")
    return "\n".join(lines)


def main(dataset_name: str, runs: list[str], score_names: list[str], top_n: int = 10, output_path: str | None = None) -> str:
    load_config()
    langfuse_client = get_langfuse_client()

    results_by_run = {run: _load_run_results(langfuse_client, dataset_name, run, score_names) for run in runs}
    common_item_ids = _get_common_item_ids(results_by_run)
    comparisons = _build_item_comparisons(runs, common_item_ids, results_by_run)

    analysis = {}
    for score_name in score_names:
        rows = _build_score_item_rows(comparisons, runs, score_name)
        analysis[score_name] = {
            "summaries": _compute_run_score_summaries(comparisons, runs, score_name),
            "worst_rows": _get_worst_rows(rows),
            "diff_rows": _get_diff_rows(rows),
        }

    markdown = _render_markdown(dataset_name, runs, score_names, top_n, analysis)
    if output_path is None:
        out_dir = PACKAGE_ROOT / "out" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    Path(output_path).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Rapport généré : {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare plusieurs runs sur plusieurs scores.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--score_names", nargs="+", required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()
    if len(args.runs) < 1:
        raise ValueError("Il faut au moins 1 run.")
    main(args.dataset_name, args.runs, args.score_names, args.top_n, args.output_path)
