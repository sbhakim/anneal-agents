#!/usr/bin/env python3
"""Bootstrap CIs + paired bootstrap tests for ANNEAL stress runs.

Addresses Reviewer nKKD: "Could you provide a stricter evaluation protocol
featuring larger task horizons, consistent multi-seed settings across all
main tables, and statistical significance testing for the key
success-rate differences?"

Two analyses:

  1. Per-(agent, scenario) bootstrap 95% CIs on:
       - success_rate
       - terminal_rfr
       - patches_accepted
       - target_holdout_failure_rate  (stress runs only)

  2. Paired bootstrap test of ANNEAL vs each baseline on
     target_holdout_failure_rate. Pairing is by seed: ANNEAL-seed-S minus
     baseline-seed-S, then bootstrap over seed-pair indices.

Inputs:
  Either a stress summary CSV (preferred — has holdout_failure_rate) OR a
  directory of metrics.json files (falls back to summary fields only).

Usage:
  conda run -n hysym python analysis/significance_analysis.py \\
    --stress-summary data/ecommerce_stress_3seed_v2/ecommerce_stress_summary.csv \\
    --output-dir data/results_rebuttal/significance/ecommerce_stress

  conda run -n hysym python analysis/significance_analysis.py \\
    --stress-summary data/travel_stress_3seed/adaptation_stress_summary.csv \\
    --output-dir data/results_rebuttal/significance/travel_stress
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# 10000 bootstrap resamples is the published convention; with 3 seeds
# the CI width is dominated by sample size, not resample count, so this
# is plenty.
N_BOOTSTRAP = 10000
CI_LOW = 2.5
CI_HIGH = 97.5

# Reference agent for paired tests (case-insensitive). We use the canonical
# project key "anneal"; runs written with "selfevolve" alias work via the
# same comparator after normalisation.
REFERENCE_AGENT = "anneal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stress-summary",
        required=True,
        help="Path to a stress summary CSV "
             "(ecommerce_stress_summary.csv or adaptation_stress_summary.csv).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def load_stress_summary(path: Path) -> List[Dict[str, Any]]:
    """Load stress summary CSV. Normalize agent name, coerce numeric fields."""
    rows: List[Dict[str, Any]] = []
    numeric_fields = (
        "success_rate",
        "holdout_success_rate",
        "accepted_patches",
        "target_holdout_failure_rate",
        "constraint_terminal_rfr",
        "patchable_terminal_rfr",
        "target_failures_prefix",
        "target_failures_pre_holdout",
        "target_failures_holdout",
        "total_tasks",
    )
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent_raw = row.get("agent", "").strip().lower()
            # Normalize the legacy "selfevolve" alias to canonical "anneal"
            row["agent_norm"] = "anneal" if agent_raw in {"selfevolve", "anneal"} else agent_raw
            for k in numeric_fields:
                if k in row and row[k] not in (None, ""):
                    try:
                        row[k] = float(row[k])
                    except (TypeError, ValueError):
                        row[k] = None
                else:
                    row[k] = None
            try:
                row["seed"] = int(row["seed"]) if row.get("seed") not in (None, "") else None
            except (TypeError, ValueError):
                row["seed"] = None
            rows.append(row)
    return rows


def bootstrap_ci(
    values: List[float],
    rng: random.Random,
    n_bootstrap: int,
) -> Dict[str, float]:
    """Percentile bootstrap 95% CI for the mean of `values`."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if n == 1:
        return {"mean": clean[0], "ci_low": clean[0], "ci_high": clean[0], "n": 1}
    means = []
    for _ in range(n_bootstrap):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(CI_LOW / 100.0 * n_bootstrap)]
    high = means[int(CI_HIGH / 100.0 * n_bootstrap)]
    return {
        "mean": sum(clean) / n,
        "ci_low": low,
        "ci_high": high,
        "n": n,
    }


def paired_bootstrap(
    paired_diffs: List[float],
    rng: random.Random,
    n_bootstrap: int,
) -> Dict[str, float]:
    """Paired bootstrap on per-seed differences (ANNEAL - baseline).

    Returns mean diff, 95% CI, and a two-sided p-value approximation: the
    fraction of resamples whose mean has opposite sign of the observed
    mean (i.e., consistent with the null of no difference). This is a
    coarse but defensible p-value for tiny seed counts.
    """
    clean = [float(d) for d in paired_diffs if d is not None and not (isinstance(d, float) and math.isnan(d))]
    n = len(clean)
    if n == 0:
        return {
            "diff_mean": float("nan"),
            "diff_ci_low": float("nan"),
            "diff_ci_high": float("nan"),
            "p_two_sided": float("nan"),
            "n_pairs": 0,
        }
    if n == 1:
        return {
            "diff_mean": clean[0],
            "diff_ci_low": clean[0],
            "diff_ci_high": clean[0],
            "p_two_sided": float("nan"),
            "n_pairs": 1,
        }
    observed_mean = sum(clean) / n
    # Centre the resampled distribution under the null by recentering on 0.
    centered = [d - observed_mean for d in clean]
    null_means = []
    for _ in range(n_bootstrap):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        null_means.append(sum(sample) / n)
    if observed_mean >= 0:
        p_one = sum(1 for m in null_means if m >= observed_mean) / n_bootstrap
    else:
        p_one = sum(1 for m in null_means if m <= observed_mean) / n_bootstrap
    p_two = min(1.0, 2.0 * p_one)
    # CI of the observed difference (not the null distribution).
    diff_resamples = []
    for _ in range(n_bootstrap):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        diff_resamples.append(sum(sample) / n)
    diff_resamples.sort()
    low = diff_resamples[int(CI_LOW / 100.0 * n_bootstrap)]
    high = diff_resamples[int(CI_HIGH / 100.0 * n_bootstrap)]
    return {
        "diff_mean": observed_mean,
        "diff_ci_low": low,
        "diff_ci_high": high,
        "p_two_sided": p_two,
        "n_pairs": n,
    }


def per_agent_cis(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    n_bootstrap: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["agent_norm"]].append(r)

    metrics_of_interest = [
        "success_rate",
        "holdout_success_rate",
        "target_holdout_failure_rate",
        "accepted_patches",
    ]
    out_rows: List[Dict[str, Any]] = []
    for agent in sorted(grouped):
        agent_rows = grouped[agent]
        out: Dict[str, Any] = {"agent": agent, "n_seeds": len(agent_rows)}
        for m in metrics_of_interest:
            values = [r[m] for r in agent_rows]
            ci = bootstrap_ci(values, rng, n_bootstrap)
            out[f"{m}_mean"] = ci["mean"]
            out[f"{m}_ci_low"] = ci["ci_low"]
            out[f"{m}_ci_high"] = ci["ci_high"]
        out_rows.append(out)
    return out_rows


def paired_anneal_vs_baselines(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    n_bootstrap: int,
) -> List[Dict[str, Any]]:
    by_agent_seed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        seed = r.get("seed")
        if seed is None:
            continue
        by_agent_seed[(r["agent_norm"], seed)] = r

    anneal_seeds = {seed for (agent, seed) in by_agent_seed if agent == REFERENCE_AGENT}
    out: List[Dict[str, Any]] = []
    for agent in sorted({a for (a, _) in by_agent_seed}):
        if agent == REFERENCE_AGENT:
            continue
        baseline_seeds = {seed for (a, seed) in by_agent_seed if a == agent}
        shared_seeds = sorted(anneal_seeds & baseline_seeds)
        diffs_holdout = []
        diffs_sr = []
        for s in shared_seeds:
            anneal_row = by_agent_seed[(REFERENCE_AGENT, s)]
            base_row = by_agent_seed[(agent, s)]
            a_hold = anneal_row.get("target_holdout_failure_rate")
            b_hold = base_row.get("target_holdout_failure_rate")
            a_sr = anneal_row.get("success_rate")
            b_sr = base_row.get("success_rate")
            if a_hold is not None and b_hold is not None:
                diffs_holdout.append(a_hold - b_hold)
            if a_sr is not None and b_sr is not None:
                diffs_sr.append(a_sr - b_sr)

        hold_test = paired_bootstrap(diffs_holdout, rng, n_bootstrap)
        sr_test = paired_bootstrap(diffs_sr, rng, n_bootstrap)
        out.append({
            "comparator": agent,
            "reference": REFERENCE_AGENT,
            "n_shared_seeds": len(shared_seeds),
            # Holdout failure rate: ANNEAL - baseline (negative = ANNEAL better)
            "holdout_diff_mean": hold_test["diff_mean"],
            "holdout_diff_ci_low": hold_test["diff_ci_low"],
            "holdout_diff_ci_high": hold_test["diff_ci_high"],
            "holdout_p_two_sided": hold_test["p_two_sided"],
            # Success rate: ANNEAL - baseline (positive = ANNEAL better)
            "sr_diff_mean": sr_test["diff_mean"],
            "sr_diff_ci_low": sr_test["diff_ci_low"],
            "sr_diff_ci_high": sr_test["diff_ci_high"],
            "sr_p_two_sided": sr_test["p_two_sided"],
        })
    return out


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
    summary_path = Path(args.stress_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"ERROR: stress summary not found: {summary_path}")
        return 1

    rows = load_stress_summary(summary_path)
    if not rows:
        print(f"ERROR: no rows parsed from {summary_path}")
        return 1

    rng = random.Random(args.seed)

    agent_cis = per_agent_cis(rows, rng, args.n_bootstrap)
    paired = paired_anneal_vs_baselines(rows, rng, args.n_bootstrap)

    write_csv(output_dir / "agent_bootstrap_cis.csv", agent_cis)
    write_csv(output_dir / "paired_anneal_vs_baselines.csv", paired)

    print(f"Source: {summary_path}")
    print(f"Output: {output_dir}/")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print()
    print("Per-agent 95% bootstrap CIs (percentile method):")
    print(
        f"{'agent':<14} {'n':>3} "
        f"{'SR_mean':>9} {'SR_CI':>20} "
        f"{'HoldFail_mean':>15} {'HoldFail_CI':>22} "
        f"{'Patches_mean':>14}"
    )
    for r in agent_cis:
        sr_ci = f"[{r['success_rate_ci_low']:.3f}, {r['success_rate_ci_high']:.3f}]"
        hf_ci = f"[{r['target_holdout_failure_rate_ci_low']:.3f}, {r['target_holdout_failure_rate_ci_high']:.3f}]"
        print(
            f"{r['agent']:<14} {r['n_seeds']:>3} "
            f"{r['success_rate_mean']:>9.3f} {sr_ci:>20} "
            f"{r['target_holdout_failure_rate_mean']:>15.3f} {hf_ci:>22} "
            f"{r['accepted_patches_mean']:>14.3f}"
        )
    print()
    print("Paired bootstrap tests (reference = anneal). Negative diff on")
    print("holdout failure rate means ANNEAL suppresses recurrence better.")
    print(
        f"{'baseline':<14} {'n':>3} "
        f"{'Δholdout':>10} {'Δhold_CI':>22} {'p2':>6}   "
        f"{'ΔSR':>8} {'ΔSR_CI':>22} {'p2':>6}"
    )
    for r in paired:
        h_ci = f"[{r['holdout_diff_ci_low']:.3f}, {r['holdout_diff_ci_high']:.3f}]"
        s_ci = f"[{r['sr_diff_ci_low']:.3f}, {r['sr_diff_ci_high']:.3f}]"
        print(
            f"{r['comparator']:<14} {r['n_shared_seeds']:>3} "
            f"{r['holdout_diff_mean']:>10.3f} {h_ci:>22} {r['holdout_p_two_sided']:>6.3f}   "
            f"{r['sr_diff_mean']:>8.3f} {s_ci:>22} {r['sr_p_two_sided']:>6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
