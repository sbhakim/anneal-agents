#!/usr/bin/env python3
"""
E-commerce recurring-failure adaptation stress test.

Mirrors run_adaptation_stress.py (travel/BookFlight) but targets
PlaceOrder:tool_schema_drift in the e-commerce domain.

Design:
  Tasks 0-5  (prefix):  PlaceOrder-heavy; tool_schema_drift active.
             SelfEvolve detects recurrence and commits an UPDATE_TOOL_SCHEMA patch.
  Tasks 6-11 (holdout): Same PlaceOrder tasks reintroduced.
             SelfEvolve: 0% PlaceOrder failures (patch suppresses injector).
             ReAct/Reflexion: ~100% PlaceOrder failures (no patch committed).

This directly produces the Table 3 equivalent for e-commerce.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


# ---------------------------------------------------------------------------
# Stress task set — all PlaceOrder-dominant so tool_schema_drift fires
# Tasks 0-5: prefix (establish failure, trigger FDKA)
# Tasks 6-11: holdout (validate persistent repair)
# ---------------------------------------------------------------------------
ECOMMERCE_STRESS_TASKS = [
    # --- PREFIX (0-7): enough failures for FDKA to detect recurrence and patch ---
    "Place an order for 1 laptop for standard customer",
    "Place an order for 2 phones for premium_member customer",
    "Validate order for tablet with payment method credit_card",
    "Place an order for 1 headphones for standard customer",
    "Validate order for laptop with payment method paypal",
    "Place an order for 2 monitors for bulk_buyer customer",
    "Place an order for 1 phone for standard customer",
    "Validate order for keyboard with payment method credit_card",
    # --- HOLDOUT (8-13): SelfEvolve patched → 0%; baselines → ~100% ---
    "Place an order for 1 keyboard for standard customer",
    "Place an order for 5 tablets for premium_member customer",
    "Validate order for phone with payment method credit_card",
    "Place an order for 1 laptop for employee customer",
    "Validate order for headphones with payment method paypal",
    "Place an order for 3 monitors for standard customer",
]

# Target failure class to track across prefix / holdout split.
# tool_schema_drift surfaces as ToolError (not PreconditionUnmet) because
# ApiSchemaDeprecated bypasses Verify-Before-Act (no precondition violation).
TARGET_FAILURE_KEY = "PlaceOrder:ToolError"
TARGET_FAILURE_PREFIX = "PlaceOrder"   # catches all PlaceOrder failure variants

HOLDOUT_START = 8   # tasks[8:] are holdout tasks — prefix now 8 tasks to give FDKA time to patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument(
        "--agents", nargs="+",
        default=["anneal", "react", "reflexion"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--mode", default="eval", choices=["demo", "eval"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(
    base: Dict[str, Any],
    agent: str,
    seed: int,
    results_dir: Path,
    log_file: Path,
) -> Dict[str, Any]:
    cfg = deepcopy(base)
    cfg.setdefault("system", {})
    cfg.setdefault("scenario", {})
    cfg.setdefault("output", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("experiment", {})

    # Domain must match scenario
    cfg["system"]["domain"] = "ecommerce"

    cfg["experiment"]["agent_to_run"] = agent
    cfg["scenario"]["name"] = "ecommerce"
    cfg["scenario"]["difficulty"] = "normal"
    cfg["scenario"]["num_tasks"] = len(ECOMMERCE_STRESS_TASKS)
    cfg["scenario"]["task_generation_seed"] = int(seed)
    cfg["scenario"]["failure_injector_seed"] = int(seed) + 101
    # High failure rate ensures tool_schema_drift fires in the prefix
    cfg["scenario"]["failure_rate"] = 0.9
    cfg["scenario"]["min_failures_in_prefix"] = 4
    cfg["scenario"]["prefix_len"] = HOLDOUT_START
    # Override tasks with PlaceOrder-dominant stress set
    cfg["scenario"]["task_overrides"] = list(ECOMMERCE_STRESS_TASKS)
    # Force PlaceOrder to always inject tool_schema_drift (controlled experiment).
    # Eliminates price_changed/policy_violation failures that the classifier
    # treats as immutable constraints, blocking FDKA.  Mirrors how the travel
    # stress test controls BookFlight:API_drift to a single failure class.
    cfg["scenario"]["placeorder_force_mode"] = "tool_schema_drift"

    cfg["output"]["results_dir"] = str(results_dir)
    cfg["logging"]["log_file"] = str(log_file)
    return cfg


def run_one(config_path: Path, mode: str, repo_root: Path) -> None:
    cmd = [sys.executable, "main.py", "--config", str(config_path), "--mode", mode]
    subprocess.run(cmd, check=True, cwd=repo_root)


def summarize_run(run_root: Path) -> Dict[str, Any]:
    metrics_path = run_root / "results" / "metrics.json"
    if not metrics_path.exists():
        return {}

    with metrics_path.open() as f:
        metrics = json.load(f)

    patches_path = run_root / "results" / "patches_committed.json"
    committed: List[Dict[str, Any]] = []
    if patches_path.exists():
        with patches_path.open() as f:
            committed = json.load(f)

    summary = metrics.get("summary", {})
    tasks = metrics.get("tasks", [])

    holdout_tasks = [t for t in tasks if int(t.get("task_id", -1)) >= HOLDOUT_START]
    prefix_tasks  = [t for t in tasks if 0 <= int(t.get("task_id", -1)) < HOLDOUT_START]

    def count_target(items: Iterable[Dict[str, Any]]) -> int:
        total = 0
        for item in items:
            observed = item.get("observed_failure_keys") or []
            if any(
                key == TARGET_FAILURE_KEY or str(key).startswith(TARGET_FAILURE_PREFIX)
                for key in observed
            ):
                total += 1
        return total

    def success_rate(items: Iterable[Dict[str, Any]]) -> float:
        lst = list(items)
        if not lst:
            return 0.0
        return sum(1 for t in lst if t.get("success")) / len(lst)

    first_patch_task = committed[0].get("task_id") if committed else None
    holdout_size = len(holdout_tasks)
    holdout_target_failures = count_target(holdout_tasks)

    return {
        "agent":                       run_root.parent.name,
        "seed":                        int(run_root.name.replace("seed_", "")),
        "metrics_json":                str(metrics_path),
        "total_tasks":                 summary.get("total_tasks", 0),
        "success_rate":                summary.get("success_rate", 0.0),
        "accepted_patches":            summary.get("patches_accepted", 0),
        "target_tta":                  (summary.get("time_to_adapt", {}) or {}).get(TARGET_FAILURE_KEY),
        "first_patch_task":            first_patch_task,
        "constraint_terminal_rfr":     summary.get("constraint_terminal_rfr", 0.0),
        "patchable_terminal_rfr":      summary.get("patchable_terminal_rfr", 0.0),
        "target_failures_prefix":      count_target(prefix_tasks),
        "target_failures_holdout":     holdout_target_failures,
        "target_holdout_failure_rate": (
            holdout_target_failures / holdout_size if holdout_size else 0.0
        ),
        "holdout_success_rate":        success_rate(holdout_tasks),
    }


def write_summary(rows: List[Dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path  = output_root / "ecommerce_stress_summary.csv"
    json_path = output_root / "ecommerce_stress_summary.json"

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "agent", "seed", "total_tasks", "success_rate",
        "holdout_success_rate", "accepted_patches",
        "target_tta", "first_patch_task",
        "constraint_terminal_rfr", "patchable_terminal_rfr",
        "target_failures_prefix", "target_failures_holdout",
        "target_holdout_failure_rate", "metrics_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n" + "=" * 70)
    print("E-COMMERCE STRESS TEST RESULTS")
    print("=" * 70)
    print(f"{'Agent':<12} {'SR':>6} {'Holdout SR':>10} {'Holdout target fail':>20} {'Patches':>8} {'TTA':>6}")
    print("-" * 70)
    for row in rows:
        print(
            f"{row['agent']:<12} "
            f"{row['success_rate']:>6.1%} "
            f"{row['holdout_success_rate']:>10.1%} "
            f"{row['target_holdout_failure_rate']:>20.1%} "
            f"{row['accepted_patches']:>8} "
            f"{str(row['target_tta']):>6}"
        )
    print("=" * 70)
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (
        (repo_root / args.base_config).resolve()
        if not Path(args.base_config).is_absolute()
        else Path(args.base_config)
    )
    output_root = Path(args.output_root)

    if not args.summarize_only:
        base_cfg = load_yaml(base_config_path)
        for agent in args.agents:
            for seed in args.seeds:
                run_root    = output_root / agent / f"seed_{seed}"
                metrics_path = run_root / "results" / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    print(f"SKIP {agent}/seed_{seed}: {metrics_path}")
                    continue

                cfg = build_config(
                    base_cfg,
                    agent=agent,
                    seed=seed,
                    results_dir=run_root / "results",
                    log_file=run_root / "logs" / f"{agent}.log",
                )
                cfg_path = run_root / "config.generated.yaml"
                dump_yaml(cfg_path, cfg)
                print(f"\nRUN {agent}/seed_{seed}")
                run_one(cfg_path, args.mode, repo_root)

    rows: List[Dict[str, Any]] = []
    for agent in args.agents:
        for seed in args.seeds:
            row = summarize_run(output_root / agent / f"seed_{seed}")
            if row:
                rows.append(row)

    if rows:
        write_summary(rows, output_root)
    else:
        print("No results found to summarize.")


if __name__ == "__main__":
    main()
