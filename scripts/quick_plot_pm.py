#!/usr/bin/env python3
"""Plot a curated set of NCU PM metrics from an .ncu-rep report.

Usage:
    python scripts/quick_plot_pm.py [report.ncu-rep]

By default the script saves a PNG next to the report.
With --interactive, it opens a Qt window instead.
"""

import argparse
import csv
import io
import subprocess
from dataclasses import dataclass

import matplotlib
import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    name: str
    metric: str
    kind: str
    category: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedMetricSpec:
    name: str
    metric: str
    kind: str
    category: str


@dataclass(frozen=True)
class MetricSeries:
    name: str
    metric: str
    category: str
    unit: str
    values: tuple[float, ...]


CATEGORY_ORDER = [
    "Overview",
    "SM",
    "L1",
    "L2",
    "DRAM",
    "Interconnect",
]


KIND_SUFFIXES = {
    "pct_peak": [".avg.pct_of_peak_sustained_elapsed"],
    "pct": [".pct", ".avg.pct_of_peak_sustained_elapsed"],
    "avg": [".avg"],
    "sum": [".sum"],
    "avg_per_second": [".avg.per_second"],
    "sum_per_second": [".sum.per_second"],
    "avg_per_cycle_active": [".avg.per_cycle_active"],
    "avg_per_cycle_elapsed": [".avg.per_cycle_elapsed"],
    "sum_per_cycle_elapsed": [".sum.per_cycle_elapsed"],
}


# Curated from scripts/ncu-metrics.txt, with a few corrections against
# `ncu --query-metrics --chip gb100`:
# - Blocks launched uses `gr__ctas_launched_realtime`
# - SM active cycles uses `sm__cycles_active`
# - L2 throughput for GCC requests uses `lts__t_sector_throughput_srcunit_gcc`
# - C2C throughput uses `ctc__throughput`
# - NVLink RX metrics use the `NVLRX` domain
CURATED_METRICS = [
    MetricSpec("Blocks Launched", "FE_B.TriageCompute.gr__ctas_launched_realtime", "sum_per_cycle_elapsed", "Overview"),
    MetricSpec("Average Blocks Active", "TPC.TriageCompute.tpc__ctas_active_realtime", "avg_per_cycle_elapsed", "Overview"),
    MetricSpec("Total Blocks Active", "TPC.TriageCompute.tpc__ctas_active_realtime", "sum_per_cycle_elapsed", "Overview"),
    MetricSpec("Average CGAs Active", "GPC_B.TriageCompute.gpc__cgas_active_realtime", "avg_per_cycle_elapsed", "Overview"),
    MetricSpec("Total CGAs Active", "GPC_B.TriageCompute.gpc__cgas_active_realtime", "sum_per_cycle_elapsed", "Overview"),
    MetricSpec("SM Active Cycles", "TPC.TriageCompute.sm__cycles_active", "avg", "SM"),
    MetricSpec("Executed IPC Active", "TPC.TriageCompute.sm__inst_executed_realtime", "avg_per_cycle_active", "SM"),
    MetricSpec("Executed IPC Elapsed", "TPC.TriageCompute.sm__inst_executed_realtime", "avg_per_cycle_elapsed", "SM"),
    MetricSpec("SM Throughput", "TPC.TriageCompute.sm__inst_executed_realtime", "pct_peak", "SM"),
    MetricSpec("SM ALU Pipe Throughput", "TPC.TriageCompute.sm__inst_executed_pipe_alu_realtime", "pct_peak", "SM"),
    MetricSpec("SM FMA Pipe Throughput", "TPC.TriageCompute.sm__pipe_fma_cycles_active_realtime", "pct_peak", "SM"),
    MetricSpec("SM FMA Heavy Pipe Throughput", "TPC.TriageCompute.sm__pipe_fmaheavy_cycles_active_realtime", "pct_peak", "SM"),
    MetricSpec("SM FMA Light Pipe Throughput", "TPC.TriageCompute.sm__pipe_fmalite_cycles_active_realtime", "pct_peak", "SM"),
    MetricSpec("SM Tensor Pipe Throughput", "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime", "pct_peak", "SM"),
    MetricSpec("SM TMEM Pipe Throughput", "SM_A.TriageCompute.sm__mem_tensor_cycles_active_realtime", "pct_peak", "SM"),
    MetricSpec("SM Uniform Pipe Throughput", "SM_A.TriageCompute.sm__inst_executed_pipe_uniform_realtime", "pct_peak", "SM"),
    MetricSpec("SM XU Pipe Throughput", "SM_A.TriageCompute.sm__inst_executed_pipe_xu_realtime", "pct_peak", "SM"),
    MetricSpec("L1 Throughput", "SM_A.TriageCompute.l1tex__throughput", "pct_peak", "L1"),
    MetricSpec("L1 Sectors", "SM_B.TriageCompute.l1tex__t_sectors", "sum", "L1"),
    MetricSpec("L1 Hit Rate", "SM_B.TriageCompute.l1tex__t_sector_hit_rate", "pct", "L1"),
    MetricSpec("L1 Lookup Hit", "SM_B.TriageCompute.l1tex__t_sectors_lookup_hit", "sum", "L1"),
    MetricSpec("L1 Lookup Miss", "SM_B.TriageCompute.l1tex__t_sectors_lookup_miss", "sum", "L1"),
    MetricSpec("L1 Wavefronts (Data)", "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts", "avg", "L1"),
    MetricSpec("L1 Wavefronts (LGDS)", "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts_mem_lgds", "avg", "L1"),
    MetricSpec("L1 Wavefronts (Shared)", "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts_mem_shared", "avg", "L1"),
    MetricSpec("L2 Throughput", "LTS.TriageCompute.lts__throughput", "pct_peak", "L2"),
    MetricSpec("L2 Throughput for L1 Requests", "LTS.TriageCompute.lts__t_sector_throughput_srcunit_tex", "pct_peak", "L2"),
    MetricSpec("L2 Throughput for GCC Requests", "LTS.TriageCompute.lts__t_sector_throughput_srcunit_gcc", "pct_peak", "L2"),
    MetricSpec("L2 Throughput to DRAM", "LTS.TriageCompute.lts__t_sector_throughput_srcnode_fbp", "pct_peak", "L2"),
    MetricSpec("SysL2 Throughput to Peer Memory", "SYSLTS.TriageCompute.syslts__t_sector_throughput_aperture_peer", "pct_peak", "L2"),
    MetricSpec("SysL2 Throughput to System Memory", "SYSLTS.TriageCompute.syslts__t_sector_throughput_aperture_sysmem", "pct_peak", "L2"),
    MetricSpec("L2 Hit Rate", "LTS.TriageCompute.lts__average_t_sector_hit_rate_realtime", "pct", "L2"),
    MetricSpec("L2 Hit Rate From L1", "LTS.TriageCompute.lts__average_t_sector_hit_rate_srcunit_tex_realtime", "pct", "L2"),
    MetricSpec("DRAM Frequency", "FBSP.TriageCompute.dram__cycles_elapsed", "avg_per_second", "DRAM"),
    MetricSpec("DRAM Throughput", "FBSP.TriageCompute.dram__throughput", "pct_peak", "DRAM"),
    MetricSpec("DRAM Read Throughput", "FBSP.TriageCompute.dram__read_throughput", "pct_peak", "DRAM"),
    MetricSpec("DRAM Write Throughput", "FBSP.TriageCompute.dram__write_throughput", "pct_peak", "DRAM"),
    MetricSpec("C2C Throughput", "TriageCompute.ctc__throughput", "pct_peak", "Interconnect", aliases=("TriageCompute.ctx__throughput",)),
    MetricSpec("NVLink Transmitted Throughput", "NVLTX.TriageCompute.nvltx__bytes", "pct_peak", "Interconnect"),
    MetricSpec("NVLink Received Throughput", "NVLRX.TriageCompute.nvlrx__bytes", "pct_peak", "Interconnect"),
    MetricSpec("NVLink Transmitted Bandwidth", "NVLTX.TriageCompute.nvltx__bytes", "sum_per_second", "Interconnect"),
    MetricSpec("NVLink Received Bandwidth", "NVLRX.TriageCompute.nvlrx__bytes", "sum_per_second", "Interconnect"),
    MetricSpec("PCIe Throughput", "PCI.TriageCompute.pcie__throughput", "pct_peak", "Interconnect"),
    MetricSpec("PCIe Read Bandwidth", "PCI.TriageCompute.pcie__read_bytes", "sum_per_second", "Interconnect"),
    MetricSpec("PCIe Write Bandwidth", "PCI.TriageCompute.pcie__write_bytes", "sum_per_second", "Interconnect"),
]


def _run_csv_command(command, timeout):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and not result.stdout:
        return None
    reader = csv.reader(io.StringIO(result.stdout))
    return list(reader)


def _query_available_metrics(chip):
    result = subprocess.run(
        ["ncu", "--query-metrics", "--chip", chip],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to query metrics for chip {chip}")

    metrics = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        token = parts[0]
        if "__" not in token:
            continue
        metrics.add(token)
    return metrics


def _metric_candidates(metric):
    candidates = [metric]
    marker = ".TriageCompute."
    if marker in metric:
        candidates.append(metric.split(marker, 1)[1])
    return candidates


def resolve_metric_specs(chip):
    available = _query_available_metrics(chip)
    resolved = []
    missing = []

    for spec in CURATED_METRICS:
        candidates = []
        for metric in (spec.metric, *spec.aliases):
            candidates.extend(_metric_candidates(metric))

        actual_metric = next((metric for metric in candidates if metric in available), None)
        if actual_metric is None:
            missing.append(spec)
            continue

        resolved.append(ResolvedMetricSpec(spec.name, actual_metric, spec.kind, spec.category))

    return resolved, missing


def _parse_metric_values(raw):
    if not raw or raw == "no data":
        return ()

    try:
        if raw.startswith("(") and raw.endswith(")"):
            rest = raw[1:-1]
            return tuple(float(v.strip().replace(",", "")) for v in rest.split(";") if v.strip())
        if " (" in raw:
            _agg, rest = raw.split(" (", 1)
            rest = rest.rstrip(")")
            return tuple(float(v.strip().replace(",", "")) for v in rest.split(";") if v.strip())
        return (float(raw.replace(",", "")),)
    except ValueError:
        return ()


def _probe_metric_series(report, metric_name):
    rows = _run_csv_command(
        [
            "ncu",
            "--import",
            report,
            "--page",
            "raw",
            "--csv",
            "--metrics",
            metric_name,
            "--print-metric-instances",
            "values",
        ],
        timeout=60,
    )
    if not rows or len(rows) < 3 or len(rows[0]) <= 11:
        return None

    header, units, row = rows[0], rows[1], rows[2]
    unit = units[11] if len(units) > 11 else ""
    raw = row[11] if len(row) > 11 else ""
    values = _parse_metric_values(raw)
    return header[11], unit, values


def collect_metric_series(report, resolved_specs):
    collected = []
    skipped = []

    for spec in resolved_specs:
        series = None
        for suffix in KIND_SUFFIXES[spec.kind]:
            probe = _probe_metric_series(report, f"{spec.metric}{suffix}")
            if probe is None:
                continue
            full_metric, unit, values = probe
            if len(values) > 1:
                series = MetricSeries(spec.name, full_metric, spec.category, unit, values)
                break

        if series is None:
            skipped.append(spec)
            continue

        collected.append(series)

    return collected, skipped


def _format_value(value):
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{value / 1e12:.2f} T"
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f} G"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.2f} K"
    if abs_value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_with_unit(value, unit):
    if not unit:
        return _format_value(value)
    return f"{_format_value(value)} {unit}"


def plot_pm(report, metrics, save=False):
    """Plot curated PM metrics as shared-x subplots in a light theme."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not metrics:
        print("No curated metrics had time-series data in the report.")
        return

    bg_fig = "#ffffff"
    bg_row = "#f6f8fb"
    text_primary = "#1f2937"
    text_secondary = "#6b7280"
    text_header = "#111827"
    grid_color = "#d7deea"
    border = "#c7d0dd"

    wave_colors = {
        "Overview": "#7c8aa5",
        "SM": "#4f87c2",
        "L1": "#2f9d8f",
        "L2": "#dd8452",
        "DRAM": "#c95d63",
        "Interconnect": "#8c6bb1",
    }

    category_rank = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    metrics = sorted(metrics, key=lambda item: (category_rank.get(item.category, 99), item.name))

    row_h = 0.55
    label_w = 3.6
    plot_w = 14.0
    fig_w = label_w + plot_w
    fig_h = row_h * len(metrics) + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg_fig)
    gs = GridSpec(
        len(metrics),
        1,
        figure=fig,
        left=label_w / fig_w,
        right=0.97,
        top=1 - 0.45 / fig_h,
        bottom=0.35 / fig_h,
        hspace=0.18,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(len(metrics))]

    prev_category = None
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = np.array(metric.values)
        x = np.arange(len(values))
        wave_color = wave_colors.get(metric.category, "#5b9bd5")

        ax.set_facecolor(bg_row)
        ax.fill_between(x, values, alpha=0.35, color=wave_color, linewidth=0)
        ax.plot(x, values, linewidth=0.8, color=wave_color)

        ax.set_xlim(0, len(values) - 1)
        if metric.unit == "%":
            ax.set_ylim(0, 100)
        else:
            ymax = np.max(values)
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

        ax.grid(True, axis="both", color=grid_color, linewidth=0.5, alpha=0.85)
        ax.tick_params(axis="both", colors=text_secondary, labelsize=6, length=0)

        if idx < len(metrics) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Sample Index", fontsize=8, color=text_secondary)

        ymin_v, ymax_v = ax.get_ylim()
        ax.set_yticks([ymin_v, ymax_v])
        ax.set_yticklabels([_format_value(ymin_v), _format_value(ymax_v)], fontsize=6, color=text_secondary)

        peak = np.max(values)
        ax.text(
            1.005,
            0.5,
            _format_with_unit(peak, metric.unit),
            transform=ax.transAxes,
            fontsize=7,
            color=text_secondary,
            va="center",
            ha="left",
            family="monospace",
        )

        for spine in ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(0.5)

        if metric.category != prev_category:
            cat_y = ax.get_position().y1 + 0.008
            fig.text(
                0.005,
                cat_y,
                f"  {metric.category}",
                fontsize=8.5,
                fontweight="bold",
                color=text_header,
                va="bottom",
                family="sans-serif",
                transform=fig.transFigure,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="#e9eef5", edgecolor="none"),
            )
        prev_category = metric.category

        label_y = (ax.get_position().y0 + ax.get_position().y1) / 2
        fig.text(
            label_w / fig_w - 0.012,
            label_y,
            metric.name,
            fontsize=7.5,
            color=text_primary,
            va="center",
            ha="right",
            family="sans-serif",
            transform=fig.transFigure,
        )

    fig.text(
        0.5,
        1 - 0.15 / fig_h,
        f"PM Sampling - {report}",
        fontsize=11,
        fontweight="bold",
        color=text_header,
        ha="center",
        va="top",
        family="sans-serif",
        transform=fig.transFigure,
    )

    if save:
        out_path = report.replace(".ncu-rep", ".pm_sampling.png")
        fig.savefig(out_path, dpi=150, facecolor=bg_fig, bbox_inches="tight", pad_inches=0.2)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="NCU PM Sampling plotter")
    parser.add_argument("report", nargs="?", default="mega-moe-kk.3.ncu-rep", help="Path to .ncu-rep file")
    parser.add_argument("--chip", default="gb100", help="Chip name used for `ncu --query-metrics`")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive Qt window instead of saving a PNG")
    args = parser.parse_args()

    if args.interactive:
        matplotlib.use("QtAgg")
    else:
        matplotlib.use("Agg")

    resolved_specs, missing_specs = resolve_metric_specs(args.chip)
    if missing_specs:
        print(f"Skipped {len(missing_specs)} curated metrics not available on {args.chip}.")
        for spec in missing_specs:
            print(f"  missing: {spec.name} -> {spec.metric}")

    metric_series, skipped_specs = collect_metric_series(args.report, resolved_specs)
    if skipped_specs:
        print(f"Skipped {len(skipped_specs)} curated metrics with no time-series data in {args.report}.")
        for spec in skipped_specs:
            print(f"  no series: {spec.name} -> {spec.metric}")

    plot_pm(args.report, metric_series, save=not args.interactive)


if __name__ == "__main__":
    main()
