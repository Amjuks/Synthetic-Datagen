from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.io_utils import ensure_parent


def _top_values(df: pd.DataFrame, column: str, limit: int = 10) -> list[dict[str, Any]]:
    if column not in df.columns or df.empty:
        return []
    vc = df[column].astype(str).value_counts(dropna=False).head(limit)
    out: list[dict[str, Any]] = []
    for key, count in vc.items():
        out.append({"value": key, "count": int(count), "pct": round(float(count) / len(df) * 100, 2)})
    return out


def _text_column_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns or df.empty:
        return {}
    s = df[column].fillna("").astype(str)
    lens = s.str.len()
    non_empty = int((lens > 0).sum())
    return {
        "non_empty_rows": non_empty,
        "coverage_pct": round(non_empty / max(1, len(df)) * 100, 2),
        "avg_chars": round(float(lens.mean()), 2),
        "median_chars": float(lens.median()),
        "p95_chars": float(lens.quantile(0.95)),
    }


def _dataset_stats(name: str, df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_cells": int(df.isna().sum().sum()) if not df.empty else 0,
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }

    text_candidates = [c for c in df.columns if c.startswith("instruction") or c.startswith("response") or c.startswith("reasoning")]
    stats["text_columns"] = text_candidates
    stats["text_stats"] = {col: _text_column_stats(df, col) for col in text_candidates}

    key_categorical = [
        "languages",
        "task_types",
        "context_types",
        "difficulty_levels",
        "topics",
        "problem_types",
        "tag_category",
        "tag_language",
        "tag_difficulty",
        "turn_count",
    ]
    stats["top_distributions"] = {col: _top_values(df, col) for col in key_categorical if col in df.columns}

    instruction_cols = [c for c in df.columns if c.startswith("instruction")]
    response_cols = [c for c in df.columns if c.startswith("response")]
    reasoning_cols = [c for c in df.columns if c.startswith("reasoning")]
    stats["conversation"] = {
        "instruction_columns": instruction_cols,
        "response_columns": response_cols,
        "reasoning_columns": reasoning_cols,
        "max_turn_columns": max(len(instruction_cols), len(response_cols)),
        "reasoning_enabled_like": len(reasoning_cols) > 0,
    }

    if "instruction" in df.columns:
        stats["instruction_unique"] = int(df["instruction"].astype(str).nunique(dropna=False))
    elif "instruction_1" in df.columns:
        stats["instruction_1_unique"] = int(df["instruction_1"].astype(str).nunique(dropna=False))

    return stats


def _render_markdown(report: dict[str, Any]) -> str:
    def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
        out = []
        out.append("| " + " | ".join(headers) + " |")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            out.append("| " + " | ".join(str(v) for v in row) + " |")
        return out

    lines: list[str] = []
    lines.append(f"# Dataset Report: {report['base_name']}")
    lines.append("")
    lines.extend(
        _table(
            ["Field", "Value"],
            [
                ["Generated at (UTC)", report["generated_at_utc"]],
                ["Config", f"`{report['config_path']}`"],
                ["Samples requested", report["samples_requested"]],
            ],
        )
    )
    lines.append("")
    lines.append("## Pipeline Summary")
    lines.append("")
    lines.extend(
        _table(
            ["Raw Rows", "Tagged Rows", "Dedup Rows", "Dedup Removed", "Dedup Removed %"],
            [[
                report["pipeline"]["raw_rows"],
                report["pipeline"]["tagged_rows"],
                report["pipeline"]["dedup_rows"],
                report["pipeline"]["dedup_removed"],
                report["pipeline"]["dedup_removed_pct"],
            ]],
        )
    )
    lines.append("")

    for ds in report["datasets"]:
        lines.append(f"## {ds['name'].title()} Dataset")
        lines.append("")
        lines.extend(
            _table(
                ["Rows", "Columns", "Missing Cells", "Max Turn Columns", "Reasoning Columns Present"],
                [[
                    ds["rows"],
                    ds["columns"],
                    ds["missing_cells"],
                    ds["conversation"]["max_turn_columns"],
                    ds["conversation"]["reasoning_enabled_like"],
                ]],
            )
        )
        lines.append("")

        if ds["text_stats"]:
            lines.append("### Text Stats")
            lines.append("")
            rows: list[list[Any]] = []
            for col, col_stats in ds["text_stats"].items():
                if not col_stats:
                    continue
                rows.append(
                    [
                        f"`{col}`",
                        f"{col_stats['coverage_pct']}%",
                        col_stats["avg_chars"],
                        col_stats["median_chars"],
                        col_stats["p95_chars"],
                    ]
                )
            if rows:
                lines.extend(_table(["Column", "Coverage", "Avg Chars", "Median Chars", "P95 Chars"], rows))
            lines.append("")

        if ds["top_distributions"]:
            lines.append("### Top Distributions")
            lines.append("")
            rows = []
            for col, values in ds["top_distributions"].items():
                if not values:
                    continue
                pretty = ", ".join(f"{v['value']} ({v['count']}, {v['pct']}%)" for v in values[:8])
                rows.append([f"`{col}`", pretty])
            if rows:
                lines.extend(_table(["Column", "Top Values"], rows))
            lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.extend(
        _table(
            ["Artifact", "Path"],
            [
                ["Raw CSV", f"`{report['files']['raw_csv']}`"],
                ["Tagged CSV", f"`{report['files']['tagged_csv']}`"],
                ["Deduplicated CSV", f"`{report['files']['dedup_csv']}`"],
                ["Report JSON", f"`{report['files']['report_json']}`"],
                ["Report Markdown", f"`{report['files']['report_md']}`"],
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def generate_report(
    *,
    base_name: str,
    config_path: str,
    samples_requested: int,
    raw_df: pd.DataFrame,
    tagged_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    raw_csv: str,
    tagged_csv: str,
    dedup_csv: str,
    report_json: str,
    report_md: str,
) -> dict[str, Any]:
    report = {
        "base_name": base_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "samples_requested": samples_requested,
        "pipeline": {
            "raw_rows": int(len(raw_df)),
            "tagged_rows": int(len(tagged_df)),
            "dedup_rows": int(len(dedup_df)),
            "dedup_removed": int(len(tagged_df) - len(dedup_df)),
            "dedup_removed_pct": round((len(tagged_df) - len(dedup_df)) / max(1, len(tagged_df)) * 100, 2),
        },
        "datasets": [
            _dataset_stats("raw", raw_df),
            _dataset_stats("tagged", tagged_df),
            _dataset_stats("deduplicated", dedup_df),
        ],
        "files": {
            "raw_csv": raw_csv,
            "tagged_csv": tagged_csv,
            "dedup_csv": dedup_csv,
            "report_json": report_json,
            "report_md": report_md,
        },
    }

    ensure_parent(report_json)
    ensure_parent(report_md)
    Path(report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(report_md).write_text(_render_markdown(report), encoding="utf-8")
    return report
