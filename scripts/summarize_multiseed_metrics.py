#!/usr/bin/env python3
"""Aggregate multi-seed metrics into mean/std manuscript tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


FIELDS = [
    ("success_rate", "sr"),
    ("observed_rfr", "rfr_obs"),
    ("terminal_rfr", "rfr_term"),
    ("constraint_satisfaction_rate", "csr"),
    ("patches_accepted", "patches_accepted"),
    ("patches_proposed", "patches_proposed"),
    ("failure_events_observed", "failure_events_observed"),
    ("recovery_tasks", "recovery_tasks"),
    ("recovery_tasks_explicit", "recovery_tasks_explicit"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed run metrics.")
    parser.add_argument("root", help="Root directory containing metrics.json files.")
    parser.add_argument("--output-dir", default="/tmp/multiseed_summary")
    return parser.parse_args()


def iter_metrics(root: Path):
    for path in sorted(root.rglob("metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload.get("summary", {})
        meta = payload.get("run_metadata") or summary.get("run_metadata") or {}
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) >= 4:
            agent, scenario, seed_dir = rel_parts[0], rel_parts[1], rel_parts[2]
        else:
            agent = str(meta.get("baseline") or meta.get("agent") or "unknown")
            scenario = "unknown"
            seed_dir = "seed_unknown"
        seed = seed_dir.removeprefix("seed_")
        yield {
            "agent": agent,
            "scenario": scenario,
            "seed": seed,
            "path": str(path),
            "summary": summary,
        }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (mean, math.sqrt(var))


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_metrics(root))
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["agent"], row["scenario"])].append(row)

    aggregate_rows = []
    for (agent, scenario), items in sorted(grouped.items()):
        out = {
            "agent": agent,
            "scenario": scenario,
            "num_runs": len(items),
        }
        for src_key, out_key in FIELDS:
            values = [float(item["summary"].get(src_key, 0.0) or 0.0) for item in items]
            mean, std = mean_std(values)
            out[f"{out_key}_mean"] = mean
            out[f"{out_key}_std"] = std
        aggregate_rows.append(out)

    csv_path = output_dir / "multiseed_summary.csv"
    if aggregate_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    json_path = output_dir / "multiseed_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows, "aggregates": aggregate_rows}, handle, indent=2)

    print(f"Runs summarized: {len(rows)}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    for row in aggregate_rows:
        print(
            f"{row['agent']}/{row['scenario']}: "
            f"SR={row['sr_mean']:.3f}±{row['sr_std']:.3f}, "
            f"RFR_obs={row['rfr_obs_mean']:.3f}±{row['rfr_obs_std']:.3f}, "
            f"Patches={row['patches_accepted_mean']:.3f}±{row['patches_accepted_std']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
