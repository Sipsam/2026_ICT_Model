"""Build comparable result groups and publish local reports."""

import csv
import json
from collections import defaultdict
from enum import StrEnum
from html import escape
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from ..data.preprocessing.console import protect_source_data
from ..pipeline.store import digest_json
from .catalog import Run


class ReportFormat(StrEnum):
    HTML = "html"
    JSON = "json"
    CSV = "csv"


def _comparison_input(model: Run) -> dict[str, Any]:
    return {key: model[key] for key in (
        "dataset", "protocol", "code_fingerprints", "speed", "metrics", "checkpoint", "model", "python", "libraries",
    )}


def comparison_groups(runs: list[Run]) -> list[Run]:
    from ..evaluation.runner import compare_results

    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run["kind"] != "evaluation" or run["status"] != "completed":
            continue
        for model in run["models"]:
            result = _comparison_input(model)
            identity = digest_json([result["dataset"], result["protocol"], result["code_fingerprints"]])
            groups[identity].append({"run_id": run["run_id"], "model_index": model["model_index"], **result})
    return [
        {"group_id": identity, "protocol": models[0]["protocol"], **compare_results(models)}
        for identity, models in groups.items()
    ]


def report_document(runs: list[Run]) -> Run:
    return {"schema_version": 1, "runs": runs, "comparison_groups": comparison_groups(runs)}


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, allow_nan=False)


def _metric_cells(model: Run) -> tuple[Any, ...]:
    metrics, speed = model.get("metrics", {}), model.get("speed", {})
    return (metrics.get("map50"), metrics.get("map50_95"), speed.get("fps"), speed.get("latency_ms", {}).get("p95"))


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    headings = "".join(f"<th>{escape(header)}</th>" for header in headers)
    cell_rows = [
        "<tr>" + "".join(f"<td>{escape(str(value)) if value is not None else '—'}</td>" for value in row) + "</tr>"
        for row in rows
    ]
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{''.join(cell_rows)}</tbody></table>"


def _html(document: Run) -> str:
    runs = document["runs"]
    overview = _table(("Run ID", "Kind", "Status", "Started (UTC)", "Ended (UTC)", "Failure reason"), [
        (run["run_id"], run["kind"], run["status"], run.get("started_at"), run.get("finished_at"),
         run.get("failure_reason")) for run in runs
    ])
    comparisons = []
    for group in document["comparison_groups"]:
        dataset = group["dataset"]
        table = _table(("Run ID", "Model", "mAP@0.5", "mAP@0.5:0.95", "FPS", "p95 latency (ms)"), [
            (model["run_id"], model["model"], *_metric_cells(model)) for model in group["models"]
        ])
        speed = "Matching reported speed environments" if group["matching_speed_environment"] else (
            "Speed environments differ or are unavailable. Review the runtime details before comparing speed."
        )
        comparisons.append(
            f"<h3>{escape(str(dataset.get('split')))} · {escape(str(dataset.get('image_count')))} images</h3>"
            f"<p>{escape(str(dataset.get('role')))} · {speed}</p>{table}"
            f"<details><summary>Dataset and protocol</summary><pre>"
            f"{escape(json.dumps({'dataset': dataset, 'protocol': group['protocol']}, indent=2))}</pre></details>"
        )
    details = "".join(
        f"<details><summary>{escape(run['run_id'])}</summary>"
        f"<pre>{escape(json.dumps(run, indent=2))}</pre></details>" for run in runs
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>ICT run comparison</title><style>'
        'body{font:16px system-ui;max-width:78rem;margin:2rem auto;padding:0 1rem}'
        'table{border-collapse:collapse;width:100%}th,td{padding:.55rem;text-align:left;border-bottom:1px solid #8885}'
        'pre{overflow:auto;padding:1rem;background:#8881}details{margin:.8rem 0}'
        '</style><body><h1>ICT run comparison</h1><h2>Run history</h2>'
        f"{overview if runs else '<p>No saved runs.</p>'}<h2>Evaluation results</h2>"
        f"{''.join(comparisons) or '<p>No completed evaluation results.</p>'}"
        f"<h2>Run details</h2>{details}</body></html>\n"
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def write_report(workspace: Path, root: Path, format: ReportFormat, document: Run) -> Path:
    root = (workspace / root).resolve()
    protect_source_data(root, workspace)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix="report-", dir=root))
    target = directory / f"comparison.{format.value}"
    partial = target.with_suffix(".partial")
    with partial.open("x", encoding="utf-8", newline="") as stream:
        if format is ReportFormat.JSON:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        elif format is ReportFormat.CSV:
            writer = csv.writer(stream)
            writer.writerow((
                "run_id", "kind", "status", "started_at", "finished_at", "failure_reason", "model", "model_status",
                "comparison_group", "split", "map50", "map50_95", "fps", "p95_latency_ms", "run_json", "model_json",
            ))
            group_ids = {
                (model["run_id"], model["model_index"]): group["group_id"]
                for group in document["comparison_groups"] for model in group["models"]
            }
            for run in document["runs"]:
                metadata = _json({key: value for key, value in run.items() if key != "models"})
                for model in run.get("models") or [{}]:
                    group = group_ids.get((run["run_id"], model.get("model_index")))
                    row = (
                        run["run_id"], run["kind"], run["status"], run.get("started_at"), run.get("finished_at"),
                        run.get("failure_reason"), model.get("model"), model.get("status"), group,
                        run.get("dataset", {}).get("split"),
                        *(_metric_cells(model) if group else (None,) * 4), metadata, _json(model) if model else None,
                    )
                    writer.writerow([_csv_cell(value) for value in row])
        else:
            stream.write(_html(document))
    partial.replace(target)
    return target
