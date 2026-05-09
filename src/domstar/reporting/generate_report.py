"""Generate plots and a sanity-check summary from training/eval artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from domstar.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/reports/latest")

    parser.add_argument("--ranker-dir", default="")
    parser.add_argument("--ranker-eval-json", default="")
    parser.add_argument("--ranker-log-file", default="")

    parser.add_argument("--finetune-dir", default="")
    parser.add_argument("--finetune-log-file", default="")

    parser.add_argument("--base-eval-json", default="")
    parser.add_argument("--dom-eval-json", default="")
    parser.add_argument("--dom-lora-eval-json", default="")

    parser.add_argument("--kernel-comparison-json", default="")
    parser.add_argument("--extra-log-file", action="append", default=[])

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any] | list[Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_history(output_dir: str) -> list[dict[str, Any]]:
    if not output_dir:
        return []
    history_path = Path(output_dir) / "log_history.json"
    payload = load_json(history_path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def partition_history(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in history:
        if "eval_loss" in row:
            eval_rows.append(row)
        elif "loss" in row:
            train_rows.append(row)
    return train_rows, eval_rows


def numeric_pairs(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for index, row in enumerate(rows, start=1):
        x_value = row.get(x_key, row.get("epoch", index))
        y_value = row.get(y_key)
        if x_value is None or y_value is None:
            continue
        try:
            x_number = float(x_value)
            y_number = float(y_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_number) and math.isfinite(y_number):
            x_values.append(x_number)
            y_values.append(y_number)
    return x_values, y_values


def plot_training_curves(
    *,
    title: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    train_steps, train_loss = numeric_pairs(train_rows, "step", "loss")
    eval_steps, eval_loss = numeric_pairs(eval_rows, "step", "eval_loss")
    if not train_steps and not eval_steps:
        return False

    plt.figure(figsize=(9, 5))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2)
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2, marker="o")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def plot_metric_bars(
    *,
    title: str,
    metric_names: list[str],
    series: dict[str, dict[str, float]],
    output_path: Path,
) -> bool:
    labels = [label for label, values in series.items() if any(metric in values for metric in metric_names)]
    if not labels:
        return False

    width = 0.8 / max(1, len(metric_names))
    x_positions = list(range(len(labels)))

    plt.figure(figsize=(10, 5))
    for metric_index, metric_name in enumerate(metric_names):
        offsets = [position + (metric_index - (len(metric_names) - 1) / 2) * width for position in x_positions]
        values = [series[label].get(metric_name, 0.0) for label in labels]
        plt.bar(offsets, values, width=width, label=metric_name)

    plt.xticks(x_positions, labels, rotation=0)
    plt.title(title)
    plt.ylim(bottom=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def plot_kernel_summary(payload: dict[str, Any], output_dir: Path) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []

    chart_paths: list[str] = []
    latency_series: dict[str, dict[str, float]] = {}
    success_series: dict[str, dict[str, float]] = {}
    for label, values in summary.items():
        if not isinstance(values, dict):
            continue
        success_rate = 0.0
        runs = values.get("runs", 0) or 0
        success = values.get("success", 0) or 0
        if runs:
            success_rate = float(success) / float(runs)
        latency_series[label] = {
            "avg_total_seconds": float(values.get("avg_total_seconds", 0.0)),
            "avg_model_seconds_total": float(values.get("avg_model_seconds_total", 0.0)),
            "avg_dom_seconds_total": float(values.get("avg_dom_seconds_total", 0.0)),
        }
        success_series[label] = {
            "success_rate": success_rate,
            "avg_steps": float(values.get("avg_steps", 0.0)),
        }

    latency_path = output_dir / "kernel_latency.png"
    if plot_metric_bars(
        title="Kernel Task Latency",
        metric_names=["avg_total_seconds", "avg_model_seconds_total", "avg_dom_seconds_total"],
        series=latency_series,
        output_path=latency_path,
    ):
        chart_paths.append(latency_path.name)

    success_path = output_dir / "kernel_success.png"
    if plot_metric_bars(
        title="Kernel Task Success and Steps",
        metric_names=["success_rate", "avg_steps"],
        series=success_series,
        output_path=success_path,
    ):
        chart_paths.append(success_path.name)
    return chart_paths


def collect_log_issues(paths: list[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    pattern = re.compile(r"\|\s+(WARNING|ERROR)\s+\|")
    for path in paths:
        if not path:
            continue
        resolved = Path(path)
        if not resolved.exists():
            continue
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if pattern.search(line) or "Traceback" in line:
                issues.append(
                    {
                        "path": str(resolved),
                        "line": line_number,
                        "message": line.strip(),
                    }
                )
    return issues


def finite_metric(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_training_health(
    *,
    name: str,
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    train_rows, eval_rows = partition_history(history)

    train_losses = [float(row["loss"]) for row in train_rows if finite_metric(row.get("loss"))]
    eval_losses = [float(row["eval_loss"]) for row in eval_rows if finite_metric(row.get("eval_loss"))]

    if history and not train_rows and not eval_rows:
        notes.append(f"{name}: trainer history exists but has no plottable loss values.")
    if any(not finite_metric(row.get("loss")) for row in train_rows if "loss" in row):
        notes.append(f"{name}: non-finite train loss detected.")
    if any(not finite_metric(row.get("eval_loss")) for row in eval_rows if "eval_loss" in row):
        notes.append(f"{name}: non-finite eval loss detected.")
    if train_losses:
        notes.append(f"{name}: train loss range {min(train_losses):.4f} -> {max(train_losses):.4f}.")
    if eval_losses:
        notes.append(f"{name}: eval loss range {min(eval_losses):.4f} -> {max(eval_losses):.4f}.")
    if final_metrics:
        final_eval_loss = final_metrics.get("eval_loss")
        if final_eval_loss is not None:
            if finite_metric(final_eval_loss):
                notes.append(f"{name}: final eval_loss={float(final_eval_loss):.4f}.")
            else:
                notes.append(f"{name}: final eval_loss is not finite.")
    return notes


def summarize_eval_delta(
    *,
    base_metrics: dict[str, Any] | None,
    dom_metrics: dict[str, Any] | None,
    dom_lora_metrics: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if not isinstance(base_metrics, dict):
        return notes

    comparisons = [
        ("DOM wrapper vs screenshot-only", dom_metrics),
        ("DOM+LoRA vs screenshot-only", dom_lora_metrics),
    ]
    for label, candidate in comparisons:
        if not isinstance(candidate, dict):
            continue
        base_element = base_metrics.get("element_accuracy")
        candidate_element = candidate.get("element_accuracy")
        if finite_metric(base_element) and finite_metric(candidate_element):
            delta = float(candidate_element) - float(base_element)
            notes.append(f"{label}: element_accuracy delta={delta:+.4f}.")

        base_op = base_metrics.get("operation_accuracy")
        candidate_op = candidate.get("operation_accuracy")
        if finite_metric(base_op) and finite_metric(candidate_op):
            delta = float(candidate_op) - float(base_op)
            notes.append(f"{label}: operation_accuracy delta={delta:+.4f}.")
    return notes


def summarize_kernel(payload: dict[str, Any] | None) -> list[str]:
    notes: list[str] = []
    if not isinstance(payload, dict):
        return notes
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return notes

    base = summary.get("base")
    domstar = summary.get("domstar")
    if isinstance(base, dict) and isinstance(domstar, dict):
        base_runs = float(base.get("runs", 0) or 0)
        dom_runs = float(domstar.get("runs", 0) or 0)
        if base_runs and dom_runs:
            base_success = float(base.get("success", 0) or 0) / base_runs
            dom_success = float(domstar.get("success", 0) or 0) / dom_runs
            notes.append(f"Kernel success delta (domstar-base)={dom_success - base_success:+.4f}.")
        if finite_metric(base.get("avg_total_seconds")) and finite_metric(domstar.get("avg_total_seconds")):
            delta = float(domstar["avg_total_seconds"]) - float(base["avg_total_seconds"])
            notes.append(f"Kernel total latency delta (domstar-base)={delta:+.4f}s.")
    return notes


def write_report(
    *,
    output_dir: Path,
    chart_names: list[str],
    summary_sections: dict[str, list[str]],
    issue_rows: list[dict[str, Any]],
    metric_snapshots: dict[str, Any],
) -> None:
    report_lines = ["# Run Report", ""]

    for section_name, notes in summary_sections.items():
        if not notes:
            continue
        report_lines.append(f"## {section_name}")
        report_lines.append("")
        for note in notes:
            report_lines.append(f"- {note}")
        report_lines.append("")

    if issue_rows:
        report_lines.append("## Warnings And Errors")
        report_lines.append("")
        for issue in issue_rows[:25]:
            report_lines.append(
                f"- `{issue['path']}:{issue['line']}`: {issue['message']}"
            )
        if len(issue_rows) > 25:
            report_lines.append(f"- ... and {len(issue_rows) - 25} more log issues")
        report_lines.append("")

    if chart_names:
        report_lines.append("## Charts")
        report_lines.append("")
        for chart_name in chart_names:
            report_lines.append(f"![{chart_name}]({chart_name})")
            report_lines.append("")

    if metric_snapshots:
        report_lines.append("## Metric Snapshots")
        report_lines.append("")
        report_lines.append("```json")
        report_lines.append(json.dumps(metric_snapshots, indent=2))
        report_lines.append("```")
        report_lines.append("")

    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(
        {
            "sections": summary_sections,
            "issues": issue_rows,
            "metrics": metric_snapshots,
            "charts": chart_names,
        },
        indent=2,
    ), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranker_metrics = load_json(Path(args.ranker_dir) / "metrics.json") if args.ranker_dir else None
    ranker_history = load_history(args.ranker_dir)
    ranker_eval_metrics = load_json(args.ranker_eval_json)

    finetune_metrics = load_json(Path(args.finetune_dir) / "train_metrics.json") if args.finetune_dir else None
    finetune_history = load_history(args.finetune_dir)

    base_eval_metrics = load_json(args.base_eval_json)
    dom_eval_metrics = load_json(args.dom_eval_json)
    dom_lora_eval_metrics = load_json(args.dom_lora_eval_json)
    kernel_payload = load_json(args.kernel_comparison_json)

    chart_names: list[str] = []

    ranker_train_rows, ranker_eval_rows = partition_history(ranker_history)
    ranker_chart = output_dir / "ranker_training.png"
    if plot_training_curves(
        title="Ranker Training Curve",
        train_rows=ranker_train_rows,
        eval_rows=ranker_eval_rows,
        output_path=ranker_chart,
    ):
        chart_names.append(ranker_chart.name)

    finetune_train_rows, finetune_eval_rows = partition_history(finetune_history)
    finetune_chart = output_dir / "northstar_training.png"
    if plot_training_curves(
        title="Northstar LoRA Training Curve",
        train_rows=finetune_train_rows,
        eval_rows=finetune_eval_rows,
        output_path=finetune_chart,
    ):
        chart_names.append(finetune_chart.name)

    if isinstance(ranker_eval_metrics, dict):
        ranker_eval_chart = output_dir / "ranker_eval.png"
        if plot_metric_bars(
            title="Ranker Evaluation",
            metric_names=[key for key in ranker_eval_metrics.keys() if key.startswith("recall@")] + ["mrr"],
            series={"ranker": {key: float(value) for key, value in ranker_eval_metrics.items() if finite_metric(value)}},
            output_path=ranker_eval_chart,
        ):
            chart_names.append(ranker_eval_chart.name)

    northstar_series: dict[str, dict[str, float]] = {}
    for label, payload in {
        "screenshot_only": base_eval_metrics,
        "base_dom": dom_eval_metrics,
        "dom_lora": dom_lora_eval_metrics,
    }.items():
        if isinstance(payload, dict):
            northstar_series[label] = {
                key: float(value)
                for key, value in payload.items()
                if key in {"json_valid_rate", "operation_accuracy", "element_accuracy", "value_accuracy"}
                and finite_metric(value)
            }
    if northstar_series:
        northstar_chart = output_dir / "northstar_eval.png"
        if plot_metric_bars(
            title="Northstar Evaluation",
            metric_names=["json_valid_rate", "operation_accuracy", "element_accuracy", "value_accuracy"],
            series=northstar_series,
            output_path=northstar_chart,
        ):
            chart_names.append(northstar_chart.name)

    chart_names.extend(plot_kernel_summary(kernel_payload if isinstance(kernel_payload, dict) else {}, output_dir))

    log_paths = [args.ranker_log_file, args.finetune_log_file, *args.extra_log_file]
    issue_rows = collect_log_issues(log_paths)

    summary_sections = {
        "Sync Expectations": [
            "Remote H100 runs do not update your local VS Code automatically.",
            "Code changes need git push from one side and git pull on the other.",
            "Artifacts and logs are ignored by git in this repo, so they stay on the machine that created them unless you copy them back explicitly.",
        ],
        "Ranker Health": summarize_training_health(
            name="Ranker",
            history=ranker_history,
            final_metrics=ranker_metrics if isinstance(ranker_metrics, dict) else None,
        ),
        "Northstar Training Health": summarize_training_health(
            name="Northstar LoRA",
            history=finetune_history,
            final_metrics=finetune_metrics if isinstance(finetune_metrics, dict) else None,
        ),
        "Model Comparison": summarize_eval_delta(
            base_metrics=base_eval_metrics if isinstance(base_eval_metrics, dict) else None,
            dom_metrics=dom_eval_metrics if isinstance(dom_eval_metrics, dict) else None,
            dom_lora_metrics=dom_lora_eval_metrics if isinstance(dom_lora_eval_metrics, dict) else None,
        ),
        "Kernel Demo Comparison": summarize_kernel(kernel_payload if isinstance(kernel_payload, dict) else None),
        "What To Look For": [
            "Loss curves should stay finite. NaN or inf means the run is not trustworthy.",
            "A healthy ranker usually shows finite eval_loss and non-trivial recall@k.",
            "DOM grounding is only justified if element_accuracy or task success goes up enough to offset added latency.",
            "If DOM increases latency but not success, the top-k summary or ranker quality is the first thing to revisit.",
        ],
    }

    metric_snapshots = {
        "ranker_metrics": ranker_metrics,
        "ranker_eval_metrics": ranker_eval_metrics,
        "finetune_metrics": finetune_metrics,
        "base_eval_metrics": base_eval_metrics,
        "dom_eval_metrics": dom_eval_metrics,
        "dom_lora_eval_metrics": dom_lora_eval_metrics,
        "kernel_summary": kernel_payload.get("summary") if isinstance(kernel_payload, dict) else None,
    }

    write_report(
        output_dir=output_dir,
        chart_names=chart_names,
        summary_sections=summary_sections,
        issue_rows=issue_rows,
        metric_snapshots=metric_snapshots,
    )
    logger.info("Report written to %s", output_dir)


if __name__ == "__main__":
    main()
