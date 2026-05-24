#!/usr/bin/env python3
"""Recovery-source decomposition for ANNEAL stress runs.

Addresses Reviewer nKKD Q2: "As noted in the discussion, many recoveries
across domains stem from verify-before-act and local repair rather than
FDKA. Could you clarify the exact proportion of observed gains
attributable specifically to committed structural edits versus these
verification mechanisms?"

For every per-seed run, classifies each task into:
  - no_failure              : success, no observed failure event
  - recovered_pre_patch     : success with observed failure(s), task_id
                              strictly before any committed patch that
                              targets the same failure class
  - recovered_post_patch    : success with observed failure(s), task_id
                              at-or-after a committed patch targeting
                              the same failure class
  - success_no_recovery     : success with observed failure(s), but no
                              matching patch in this run (e.g., baseline)
  - terminal_failure        : task failed (success=False)

Inputs:
  --root  : a directory containing one or more metrics.json files
            (will be discovered recursively).
Outputs (under --output-dir):
  - recovery_breakdown_per_run.csv   : one row per (agent, scenario, seed)
  - recovery_breakdown_per_task.csv  : one row per task (debug)
  - recovery_breakdown_aggregate.csv : mean/std across seeds per (agent, scenario)

Usage:
  conda run -n hysym python analysis/recovery_source_breakdown.py \\
    --root data/ecommerce_stress_3seed_v2 \\
    --output-dir data/results_rebuttal/recovery_breakdown/ecommerce_stress
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CATEGORY_ORDER = [
    "no_failure",
    "recovered_pre_patch",
    "recovered_post_patch",
    "success_no_recovery",
    "terminal_failure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing metrics.json files.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-run / per-task / aggregate CSVs.")
    parser.add_argument(
        "--scheme",
        choices=["agent_seed", "agent_scenario_seed"],
        default="agent_seed",
        help=(
            "Path interpretation. 'agent_seed' expects <root>/<agent>/seed_<S>/results/metrics.json. "
            "'agent_scenario_seed' expects <root>/<agent>/<scenario>/seed_<S>/results/metrics.json."
        ),
    )
    return parser.parse_args()


def discover_runs(root: Path, scheme: str) -> List[Dict[str, Any]]:
    """Find every metrics.json under root and label by path scheme."""
    runs = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        rel = metrics_path.relative_to(root)
        parts = rel.parts
        # Expected:
        #   agent_seed:           <agent>/seed_<S>/results/metrics.json   (len 4)
        #   agent_scenario_seed:  <agent>/<scenario>/seed_<S>/results/metrics.json (len 5)
        if scheme == "agent_seed" and len(parts) >= 4:
            agent = parts[0]
            scenario = root.name  # derive from root dirname
            seed_dir = parts[1]
        elif scheme == "agent_scenario_seed" and len(parts) >= 5:
            agent = parts[0]
            scenario = parts[1]
            seed_dir = parts[2]
        else:
            # Fall back to best-effort parse
            agent = parts[0] if parts else "unknown"
            scenario = root.name
            seed_dir = next((p for p in parts if p.startswith("seed_")), "seed_unknown")

        seed = seed_dir.removeprefix("seed_")
        committed_path = metrics_path.parent / "patches_committed.json"
        runs.append({
            "agent": agent,
            "scenario": scenario,
            "seed": seed,
            "metrics_path": metrics_path,
            "committed_path": committed_path if committed_path.exists() else None,
        })
    return runs


def load_committed_patches(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else []
    except json.JSONDecodeError:
        return []


def normalize_failure_key(key: str) -> str:
    """Reduce a failure key to its operator + family prefix for matching against patches.

    Examples:
      'PlaceOrder:ToolError'             -> 'PlaceOrder'
      'BookFlight:ToolError:API-V2'      -> 'BookFlight'
    """
    return str(key).split(":")[0]


def classify_task(
    task: Dict[str, Any],
    committed_patches: List[Dict[str, Any]],
) -> str:
    """Assign a recovery-source category to a single task."""
    success = bool(task.get("success", False))
    observed_keys = list(task.get("observed_failure_keys") or [])
    task_id = int(task.get("task_id", -1))

    if not success:
        return "terminal_failure"

    if not observed_keys:
        return "no_failure"

    # Success after observed failure events: was a patch active on this task?
    observed_operators = {normalize_failure_key(k) for k in observed_keys}

    matching_patch_at_or_before = False
    matching_patch_after = False
    for patch in committed_patches:
        try:
            patch_task = int(patch.get("task_id", 1 << 30))
        except (TypeError, ValueError):
            patch_task = 1 << 30
        op = str(patch.get("operator") or "")
        if op in observed_operators:
            if patch_task <= task_id:
                matching_patch_at_or_before = True
            else:
                matching_patch_after = True

    if matching_patch_at_or_before:
        return "recovered_post_patch"
    if matching_patch_after:
        # Patch exists in this run but was committed later — this task recovered without it
        return "recovered_pre_patch"
    if committed_patches:
        # Patches exist but none target this operator family
        return "recovered_pre_patch"
    # Baseline (no patches in run) — clearer label
    return "success_no_recovery"


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    metrics = json.loads(run["metrics_path"].read_text())
    tasks = metrics.get("tasks", [])
    committed = load_committed_patches(run["committed_path"])
    summary = metrics.get("summary", {})

    counts = defaultdict(int)
    per_task_rows: List[Dict[str, Any]] = []
    for task in tasks:
        cat = classify_task(task, committed)
        counts[cat] += 1
        per_task_rows.append({
            "agent": run["agent"],
            "scenario": run["scenario"],
            "seed": run["seed"],
            "task_id": task.get("task_id"),
            "success": task.get("success"),
            "observed_failure_keys": ";".join(task.get("observed_failure_keys") or []),
            "recovered_failure_keys": ";".join(task.get("recovered_failure_keys") or []),
            "category": cat,
        })

    total = max(len(tasks), 1)
    row = {
        "agent": run["agent"],
        "scenario": run["scenario"],
        "seed": run["seed"],
        "total_tasks": len(tasks),
        "patches_committed": len(committed),
        "first_patch_task": committed[0].get("task_id") if committed else None,
        "success_rate": summary.get("success_rate"),
        "terminal_rfr": summary.get("terminal_rfr"),
        "observed_rfr": summary.get("observed_rfr"),
        "recovery_tasks_explicit": summary.get("recovery_tasks_explicit"),
    }
    for cat in CATEGORY_ORDER:
        row[f"count_{cat}"] = counts.get(cat, 0)
        row[f"frac_{cat}"] = counts.get(cat, 0) / total
    return {"summary_row": row, "per_task_rows": per_task_rows}


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (mean, math.sqrt(var))


def aggregate(per_run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(row["agent"], row["scenario"])].append(row)

    aggregates = []
    for (agent, scenario), rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "agent": agent,
            "scenario": scenario,
            "num_seeds": len(rows),
        }
        for cat in CATEGORY_ORDER:
            counts = [r[f"count_{cat}"] for r in rows]
            fracs = [r[f"frac_{cat}"] for r in rows]
            mc, sc = mean_std([float(x) for x in counts])
            mf, sf = mean_std([float(x) for x in fracs])
            out[f"{cat}_count_mean"] = mc
            out[f"{cat}_count_std"] = sc
            out[f"{cat}_frac_mean"] = mf
            out[f"{cat}_frac_std"] = sf
        patches_committed = [r["patches_committed"] for r in rows]
        mp, sp = mean_std([float(x) for x in patches_committed])
        out["patches_committed_mean"] = mp
        out["patches_committed_std"] = sp
        aggregates.append(out)
    return aggregates


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root, args.scheme)
    if not runs:
        print(f"No metrics.json files found under {root}")
        return 1

    per_run_rows: List[Dict[str, Any]] = []
    per_task_rows: List[Dict[str, Any]] = []
    for run in runs:
        result = summarize_run(run)
        per_run_rows.append(result["summary_row"])
        per_task_rows.extend(result["per_task_rows"])

    aggregate_rows = aggregate(per_run_rows)

    write_csv(output_dir / "recovery_breakdown_per_run.csv", per_run_rows)
    write_csv(output_dir / "recovery_breakdown_per_task.csv", per_task_rows)
    write_csv(output_dir / "recovery_breakdown_aggregate.csv", aggregate_rows)

    print(f"Runs processed: {len(per_run_rows)}")
    print(f"Tasks processed: {len(per_task_rows)}")
    print(f"Output: {output_dir}/")
    print()
    print("Aggregate (mean ± std across seeds):")
    print(
        f"{'agent':<14} {'scenario':<28} "
        f"{'no_fail%':>9} {'pre_patch%':>11} {'post_patch%':>12} "
        f"{'no_rec%':>9} {'fail%':>7} {'patches':>8}"
    )
    for r in aggregate_rows:
        print(
            f"{r['agent']:<14} {r['scenario']:<28} "
            f"{r['no_failure_frac_mean']*100:>8.1f}% "
            f"{r['recovered_pre_patch_frac_mean']*100:>10.1f}% "
            f"{r['recovered_post_patch_frac_mean']*100:>11.1f}% "
            f"{r['success_no_recovery_frac_mean']*100:>8.1f}% "
            f"{r['terminal_failure_frac_mean']*100:>6.1f}% "
            f"{r['patches_committed_mean']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
