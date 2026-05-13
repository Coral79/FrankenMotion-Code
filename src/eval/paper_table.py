"""Compose the paper's Table-1-style summary from the detailed CSVs produced
by ``src.eval.run``.

Paper Table 1 reports 13 numbers per method across three column groups:

    Avg-part   R@1 / R@3 / M2T
    Per-action R@1 / R@3 / M2T / FID / Div
    Per-seq    R@1 / R@3 / M2T / FID / Div

The protocols and FID variants are mixed on purpose (confirmed by cross-
checking the reference run at ``val_full_dataset_last_20251106_205756/``):

    * Avg-part   → ``guo+threshold_parts_average.csv``  columns m2t_R01 / m2t_R03 / M2T
    * Per-action → ``guo_action.csv``                   columns m2t_R01 / m2t_R03 / M2T / FID+   / Diversity
    * Per-seq    → ``guo_caption.csv``                  columns m2t_R01 / m2t_R03 / M2T / FID_nrom / Diversity

The composer reads the AVERAGE and STD rows and emits one line per method in
the paper's format (value ± 95% CI). A ``--detailed`` mode prints every
per-run number for drilling in.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from math import sqrt
from typing import Dict


# Fixed mapping — paper cell → (csv file suffix, column name).
PAPER_CELLS = {
    "avg_part_R1":  ("guo+threshold_parts_average", "m2t_R01"),
    "avg_part_R3":  ("guo+threshold_parts_average", "m2t_R03"),
    "avg_part_M2T": ("guo+threshold_parts_average", "M2T"),

    "per_action_R1":  ("guo_action", "m2t_R01"),
    "per_action_R3":  ("guo_action", "m2t_R03"),
    "per_action_M2T": ("guo_action", "M2T"),
    "per_action_FID": ("guo_action", "FID+"),
    "per_action_Div": ("guo_action", "Diversity"),

    "per_seq_R1":  ("guo_caption", "m2t_R01"),
    "per_seq_R3":  ("guo_caption", "m2t_R03"),
    "per_seq_M2T": ("guo_caption", "M2T"),
    "per_seq_FID": ("guo_caption", "FID_nrom"),
    "per_seq_Div": ("guo_caption", "Diversity"),
}


@dataclass
class Stat:
    avg: float
    std: float

    def ci95(self, n_runs: int) -> float:
        # Paper reports 95% confidence intervals (±) computed as 1.96 * std / sqrt(n).
        if n_runs <= 0:
            return 0.0
        return 1.96 * self.std / sqrt(n_runs)


def _parse_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Return ``{run_label: {metric_name: value}}`` from a detailed-eval CSV."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.pop("run")
            out[label] = {
                k: float(v) for k, v in row.items() if v not in ("", None)
            }
    return out


def _discover_csvs(experiment_dir: str, max_runs: int, dataset_name: str) -> Dict[str, str]:
    """Return ``{suffix: path}`` for all protocol/subset CSVs in ``experiment_dir``."""
    prefix_re = re.compile(rf"^evaluation_results_(?:{max_runs}_)?{re.escape(dataset_name)}_(.+)\.csv$")
    hits: Dict[str, str] = {}
    for name in os.listdir(experiment_dir):
        m = prefix_re.match(name)
        if not m:
            continue
        suffix = m.group(1)
        # Prefer the CSV with the explicit max_runs prefix if both exist
        path = os.path.join(experiment_dir, name)
        if suffix not in hits or f"_{max_runs}_" in name:
            hits[suffix] = path
    return hits


def build_table(experiment_dir: str, max_runs: int = 20, dataset_name: str = "amass-part-20") -> Dict[str, Stat]:
    """Return the dict mapping paper-cell key to ``Stat(avg, std)``."""
    csvs = _discover_csvs(experiment_dir, max_runs, dataset_name)
    results: Dict[str, Stat] = {}

    for cell_key, (suffix, col) in PAPER_CELLS.items():
        if suffix not in csvs:
            results[cell_key] = Stat(float("nan"), float("nan"))
            continue
        rows = _parse_csv(csvs[suffix])
        avg = rows.get("AVERAGE", {}).get(col, float("nan"))
        std = rows.get("STD", {}).get(col, float("nan"))
        results[cell_key] = Stat(avg, std)

    return results


def build_gt(experiment_dir: str, max_runs: int = 20, dataset_name: str = "amass-part-20") -> Dict[str, float]:
    """Return ``{paper-cell key: GT value}`` reading the GT row from each protocol-mapped CSV.

    GT is a single measurement per CSV (no AVG/STD), so we return the raw value.
    """
    csvs = _discover_csvs(experiment_dir, max_runs, dataset_name)
    gt: Dict[str, float] = {}
    for cell_key, (suffix, col) in PAPER_CELLS.items():
        if suffix not in csvs:
            gt[cell_key] = float("nan")
            continue
        rows = _parse_csv(csvs[suffix])
        gt[cell_key] = rows.get("GT", {}).get(col, float("nan"))
    return gt


def build_per_run(experiment_dir: str, max_runs: int = 20, dataset_name: str = "amass-part-20") -> Dict[str, Dict[str, float]]:
    """Return ``{run_label: {paper-cell key: value}}`` for run_000..run_<max_runs-1>.

    Pulls from the protocol-mapped CSV for each cell, mirroring the GT and AVG/STD logic.
    """
    csvs = _discover_csvs(experiment_dir, max_runs, dataset_name)
    parsed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cell_key, (suffix, col) in PAPER_CELLS.items():
        if suffix not in parsed:
            parsed[suffix] = _parse_csv(csvs[suffix]) if suffix in csvs else {}

    out: Dict[str, Dict[str, float]] = {}
    for run_idx in range(max_runs):
        label = f"run_{run_idx:03d}"
        out[label] = {}
        for cell_key, (suffix, col) in PAPER_CELLS.items():
            out[label][cell_key] = parsed[suffix].get(label, {}).get(col, float("nan"))
    return out


def format_paper_table(stats: Dict[str, Stat], n_runs: int = 20, method_name: str = "FrankenMotion",
                       gt: Dict[str, float] | None = None,
                       per_run: Dict[str, Dict[str, float]] | None = None) -> str:
    """One wide table: rows = GT / run_<i> / AVERAGE / STD / <method>+CI; columns = all 13 paper cells."""
    # (cell_key, header_label, decimals)
    cells = [
        ("avg_part_R1",   "Avg-part R@1", 2),
        ("avg_part_R3",   "Avg-part R@3", 2),
        ("avg_part_M2T",  "Avg-part M2T", 3),
        ("per_action_R1",  "Per-action R@1", 2),
        ("per_action_R3",  "Per-action R@3", 2),
        ("per_action_M2T", "Per-action M2T", 3),
        ("per_action_FID", "Per-action FID", 3),
        ("per_action_Div", "Per-action Div", 2),
        ("per_seq_R1",    "Per-seq R@1", 2),
        ("per_seq_R3",    "Per-seq R@3", 2),
        ("per_seq_M2T",   "Per-seq M2T", 3),
        ("per_seq_FID",   "Per-seq FID", 3),
        ("per_seq_Div",   "Per-seq Div", 2),
    ]

    def fmt_value(v: float, decimals: int) -> str:
        if v != v:
            return "—"
        return f"{v:.{decimals}f}"

    def fmt_ci(s: Stat, decimals: int) -> str:
        if s.avg != s.avg:
            return "—"
        return f"{s.avg:.{decimals}f}±{s.ci95(n_runs):.2f}"

    # Column widths: max of header, all body cells we'll emit.
    col_widths = []
    for k, hdr, d in cells:
        w = len(hdr)
        if gt is not None:
            w = max(w, len(fmt_value(gt[k], d)))
        if per_run is not None:
            for run_label in per_run:
                w = max(w, len(fmt_value(per_run[run_label][k], d)))
        w = max(w, len(fmt_value(stats[k].avg, d)))
        w = max(w, len(fmt_value(stats[k].std, d)))
        w = max(w, len(fmt_ci(stats[k], d)))
        col_widths.append(w + 2)  # padding

    label_w = 16
    if per_run is not None:
        label_w = max(label_w, max(len(r) for r in per_run) + 2)
    label_w = max(label_w, len(method_name) + 2)

    def row(label: str, render) -> str:
        cells_str = "".join(render(k, hdr, d).rjust(w) for (k, hdr, d), w in zip(cells, col_widths))
        return label.ljust(label_w) + cells_str

    lines = []
    lines.append(f"Paper Table 1 — {method_name} (n_runs={n_runs})")
    header = row("row", lambda k, hdr, d: hdr)
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    if gt is not None:
        lines.append(row("GT", lambda k, hdr, d: fmt_value(gt[k], d)))
    if per_run is not None:
        for run_label in sorted(per_run.keys()):
            lines.append(row(run_label, lambda k, hdr, d: fmt_value(per_run[run_label][k], d)))
    lines.append("-" * len(header))
    lines.append(row("AVERAGE", lambda k, hdr, d: fmt_value(stats[k].avg, d)))
    lines.append(row("CI95",    lambda k, hdr, d: fmt_value(stats[k].ci95(n_runs), d)))
    return "\n".join(lines)


_CSV_COLUMNS = [
    ("avg_part_R1",   "Avg-part R@1"),
    ("avg_part_R3",   "Avg-part R@3"),
    ("avg_part_M2T",  "Avg-part M2T"),
    ("per_action_R1",  "Per-action R@1"),
    ("per_action_R3",  "Per-action R@3"),
    ("per_action_M2T", "Per-action M2T"),
    ("per_action_FID", "Per-action FID"),
    ("per_action_Div", "Per-action Div"),
    ("per_seq_R1",    "Per-seq R@1"),
    ("per_seq_R3",    "Per-seq R@3"),
    ("per_seq_M2T",   "Per-seq M2T"),
    ("per_seq_FID",   "Per-seq FID"),
    ("per_seq_Div",   "Per-seq Div"),
]


def save_paper_table_csv(stats: Dict[str, Stat], gt: Dict[str, float] | None,
                         per_run: Dict[str, Dict[str, float]] | None, n_runs: int,
                         method_name: str, path: str) -> None:
    """Write the paper-Table-1 cells as a wide CSV (one row per turn + summary rows).

    Columns: row, then 13 paper-Table-1 cells in a fixed order.
    Rows: GT (if provided), run_000..run_<n-1> (if provided), AVERAGE, STD, <method>_CI95.
    """
    headers = ["row"] + [label for _, label in _CSV_COLUMNS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        if gt is not None:
            w.writerow(["GT"] + [f"{gt[k]:.6g}" for k, _ in _CSV_COLUMNS])
        if per_run is not None:
            for label in sorted(per_run.keys()):
                w.writerow([label] + [f"{per_run[label][k]:.6g}" for k, _ in _CSV_COLUMNS])
        w.writerow(["AVERAGE"] + [f"{stats[k].avg:.6g}" for k, _ in _CSV_COLUMNS])
        w.writerow(["CI95"]    + [f"{stats[k].ci95(n_runs):.6g}" for k, _ in _CSV_COLUMNS])


def _missing_protocols(experiment_dir, max_runs, dataset_name):
    """Return the list of protocols whose detailed-eval CSVs aren't on disk yet."""
    needed = ["guo", "guo+threshold"]
    csvs = _discover_csvs(experiment_dir, max_runs, dataset_name)
    have_protocols = {key.split("_", 1)[0] for key in csvs}
    return [p for p in needed if p not in have_protocols]


def _run_eval_for_protocols(experiment_dir, protocols, max_runs, dataset_name,
                            eval_model_dir, base_splits_root, base_annotations_root,
                            split, device):
    """Invoke `src.eval.run` once per protocol that's missing CSVs."""
    import subprocess
    import sys as _sys
    for protocol in protocols:
        cmd = [
            _sys.executable, "-m", "src.eval.run", experiment_dir,
            "--calculate_gt_metrics",
            "--dataset_name", dataset_name,
            "--max_runs", str(max_runs),
            "--protocol", protocol,
            "--split", split,
            "--base_splits_root", base_splits_root,
            "--base_annotations_root", base_annotations_root,
            "--device", device,
        ]
        if eval_model_dir:
            cmd.extend(["--eval_model_dir", eval_model_dir])
        print(f"\n[paper_table] auto-running: src.eval.run --protocol {protocol}")
        subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Emit paper-Table-1 summary from detailed-eval CSVs. "
                                             "Auto-runs missing protocols (guo and guo+threshold) if needed.")
    ap.add_argument("experiment_dir", help="Directory containing the generated motions (turn_<i>/data/) and evaluation_results_*.csv")
    ap.add_argument("--max_runs", type=int, default=20)
    ap.add_argument("--dataset_name", default="amass-part-20")
    ap.add_argument("--method_name", default="FrankenMotion")
    ap.add_argument("--detailed", action="store_true",
                    help="Also print the underlying per-run CSVs in a condensed form.")
    ap.add_argument("--no_gt", action="store_true",
                    help="Suppress the GT row in the output (default: GT is shown).")
    ap.add_argument("--no_per_run", action="store_true",
                    help="Suppress per-turn rows; only show AVG/STD/method (default: per-run is shown).")
    ap.add_argument("--save_csv", default=None,
                    help="Output CSV path (default: <experiment_dir>/paper_table.csv). "
                         "Pass an empty string to disable saving.")
    # Forwarded to src.eval.run if missing-protocol CSVs need to be generated:
    ap.add_argument("--eval_model_dir", default="pretrained/eval_model",
                    help="Eval-encoder directory (used when auto-running src.eval.run for missing protocols).")
    ap.add_argument("--base_splits_root", default="datasets/annotations",
                    help="Splits root (forwarded to src.eval.run).")
    ap.add_argument("--base_annotations_root", default="datasets/annotations",
                    help="Annotations root (forwarded to src.eval.run).")
    ap.add_argument("--split", default="test",
                    help="Split name (forwarded to src.eval.run).")
    ap.add_argument("--device", default="cpu",
                    help="Eval device (forwarded to src.eval.run).")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Don't auto-run src.eval.run; only read pre-existing CSVs.")
    ap.add_argument("--keep_intermediate", action="store_true",
                    help="Keep the per-protocol intermediate `evaluation_results_*.csv` + `.json` "
                         "files alongside paper_table.csv. By default they are deleted after "
                         "paper_table.csv is built to keep the experiment directory clean.")
    args = ap.parse_args()

    if not args.skip_eval:
        missing = _missing_protocols(args.experiment_dir, args.max_runs, args.dataset_name)
        if missing:
            _run_eval_for_protocols(
                args.experiment_dir, missing, args.max_runs, args.dataset_name,
                args.eval_model_dir, args.base_splits_root, args.base_annotations_root,
                args.split, args.device,
            )

    stats = build_table(args.experiment_dir, args.max_runs, args.dataset_name)
    gt = build_gt(args.experiment_dir, args.max_runs, args.dataset_name) if not args.no_gt else None
    per_run = build_per_run(args.experiment_dir, args.max_runs, args.dataset_name) if not args.no_per_run else None
    print(format_paper_table(stats, args.max_runs, args.method_name, gt=gt, per_run=per_run))

    save_path = args.save_csv if args.save_csv is not None else os.path.join(args.experiment_dir, "paper_table.csv")
    if save_path:
        save_paper_table_csv(stats, gt, per_run, args.max_runs, args.method_name, save_path)
        print(f"\n[saved] {save_path}")

    if not args.keep_intermediate:
        import glob as _glob
        removed = 0
        for pat in (
            os.path.join(args.experiment_dir, "evaluation_results_*.csv"),
            os.path.join(args.experiment_dir, "evaluation_results_*.json"),
        ):
            for p in _glob.glob(pat):
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"[cleanup] removed {removed} intermediate evaluation_results_* files (pass --keep_intermediate to retain).")

    if args.detailed:
        for suffix, path in sorted(_discover_csvs(args.experiment_dir, args.max_runs, args.dataset_name).items()):
            print(f"\n[{suffix}] {path}")
            with open(path) as f:
                print(f.read())


if __name__ == "__main__":
    main()
