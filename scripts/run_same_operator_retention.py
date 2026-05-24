#!/usr/bin/env python3
"""Same-operator capability-retention probe for rebuttal Track F-2.

Complements the existing capability-retention probe (Track F) which used
10 unrelated-operator tasks (ApplyPromoCode, CalculateShipping, etc.).
That probe answers "does the patch break adjacent operators?" but not
"does the patch break the patched operator itself for non-failing inputs?"

This script measures locality at the **patched operator**: 10 PlaceOrder
retention tasks at task IDs outside the failure-injector horizon. The
hypothesis is that the UPDATE_TOOL_SCHEMA patch updates the operator to
a valid API path, so non-injected PlaceOrder tasks should succeed both
pre-patch (original schema works for clean tasks) and post-patch (new
schema also works for clean tasks).

Outputs:
  - retention_summary.csv / .json
  - retention_tasks.csv
  - per-seed metrics.json and patches_committed.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.system import SelfEvolveSystem
from scripts.run_ecommerce_stress import ECOMMERCE_STRESS_TASKS, HOLDOUT_START


# Same-operator retention probes: all exercise PlaceOrder (the patched
# operator) with task IDs >= 1000 so the failure injector (which seeds
# task IDs 0..13) does not fire. Parameter diversity (products, customer
# tiers, quantities) probes whether the patched schema generalizes
# beyond the stress-phase calling pattern.
SAME_OP_RETENTION_TASKS = [
    "Place an order for 1 laptop for standard customer",
    "Place an order for 2 keyboards for premium_member customer",
    "Place an order for 1 monitor for standard customer",
    "Place an order for 3 headphones for bulk_buyer customer",
    "Place an order for 1 phone for standard customer",
    "Validate order for tablet with payment method credit_card",
    "Place an order for 2 tablets for premium_member customer",
    "Validate order for laptop with payment method paypal",
    "Place an order for 1 keyboard for employee customer",
    "Place an order for 4 monitors for standard customer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 99])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--provider", default=None, help="Override fdka.propose_edit.llm_provider.")
    parser.add_argument("--model", default=None, help="Override fdka.propose_edit.model.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Capture verbose runtime output per seed.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_config(
    base: Dict[str, Any],
    seed: int,
    run_root: Path,
    provider: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    cfg = deepcopy(base)
    cfg.setdefault("system", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("scenario", {})
    cfg.setdefault("output", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("governance", {}).setdefault("provenance", {})
    cfg.setdefault("fdka", {}).setdefault("propose_edit", {})

    cfg["system"]["domain"] = "ecommerce"
    cfg["experiment"]["agent_to_run"] = "anneal"
    cfg["scenario"]["name"] = "ecommerce"
    cfg["scenario"]["difficulty"] = "normal"
    cfg["scenario"]["num_tasks"] = len(ECOMMERCE_STRESS_TASKS)
    cfg["scenario"]["task_generation_seed"] = int(seed)
    cfg["scenario"]["failure_injector_seed"] = int(seed) + 101
    cfg["scenario"]["failure_rate"] = 0.9
    cfg["scenario"]["min_failures_in_prefix"] = 4
    cfg["scenario"]["prefix_len"] = HOLDOUT_START
    cfg["scenario"]["task_overrides"] = list(ECOMMERCE_STRESS_TASKS)
    cfg["scenario"]["placeorder_force_mode"] = "tool_schema_drift"

    if provider:
        cfg["fdka"]["propose_edit"]["llm_provider"] = provider
    if model:
        cfg["fdka"]["propose_edit"]["model"] = model

    cfg["output"]["results_dir"] = str(run_root / "results")
    cfg["output"]["plots_dir"] = str(run_root / "results" / "plots")
    cfg["logging"]["log_file"] = str(run_root / "logs" / "anneal.log")
    cfg["governance"]["provenance"]["log_path"] = str(run_root / "logs" / "provenance.jsonl")
    return cfg


def run_probe_phase(
    system: SelfEvolveSystem,
    phase: str,
    seed: int,
    task_id_offset: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, instruction in enumerate(SAME_OP_RETENTION_TASKS):
        task_id = task_id_offset + idx
        result = system.run_task(task_id, instruction)
        rows.append(
            {
                "seed": seed,
                "phase": phase,
                "probe_id": idx,
                "task_id": task_id,
                "instruction": instruction,
                "status": result.get("status"),
                "success": result.get("status") == "success",
                "observed_failure_events": len(
                    [
                        entry
                        for entry in result.get("trace", []) or []
                        if isinstance(entry, dict) and ("error" in entry or "error_type" in entry)
                    ]
                ),
            }
        )
    return rows


def run_stress_phase(system: SelfEvolveSystem, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_id in range(len(ECOMMERCE_STRESS_TASKS)):
        instruction = system.scenario.get_task(task_id)
        result = system.run_task(task_id, instruction)
        rows.append(
            {
                "seed": seed,
                "phase": "stress",
                "task_id": task_id,
                "instruction": instruction,
                "status": result.get("status"),
                "success": result.get("status") == "success",
            }
        )
    return rows


def save_system_artifacts(system: SelfEvolveSystem, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    system.metrics.save(results_dir / "metrics.json")
    system.experience_pool.save(results_dir / "experience_pool.json")
    write_json(results_dir / "patches_committed.json", system._committed_patches)
    try:
        write_json(results_dir / "governance_summary.json", system.guard.get_statistics())
    except Exception:
        pass
    try:
        system._write_run_bundle(results_dir)
    except Exception:
        pass


def summarize_seed(seed: int, run_root: Path, task_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pre = [r for r in task_rows if r["phase"] == "pre"]
    post = [r for r in task_rows if r["phase"] == "post"]
    pre_sr = sum(1 for r in pre if r["success"]) / len(pre) if pre else math.nan
    post_sr = sum(1 for r in post if r["success"]) / len(post) if post else math.nan

    patches_path = run_root / "results" / "patches_committed.json"
    patches = []
    if patches_path.exists():
        patches = json.loads(patches_path.read_text(encoding="utf-8"))

    return {
        "agent": "anneal",
        "domain": "ecommerce",
        "seed": seed,
        "probe_type": "same_operator_PlaceOrder",
        "num_probe_tasks": len(pre),
        "pre_success_rate": pre_sr,
        "post_success_rate": post_sr,
        "delta_post_minus_pre": post_sr - pre_sr,
        "stress_success_rate": (
            sum(1 for r in task_rows if r["phase"] == "stress" and r["success"])
            / max(1, sum(1 for r in task_rows if r["phase"] == "stress"))
        ),
        "patches_committed": len(patches),
        "patch_operators": ";".join(str(p.get("operator")) for p in patches),
        "metrics_json": str(run_root / "results" / "metrics.json"),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    base = load_yaml(base_path)
    output_root = Path(args.output_root)

    all_task_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        run_root = output_root / "anneal" / "ecommerce" / f"seed_{seed}"
        metrics_path = run_root / "results" / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"SKIP seed_{seed}: {metrics_path}")
            continue

        cfg = build_config(base, seed, run_root, provider=args.provider, model=args.model)
        write_json(run_root / "config.generated.json", cfg)
        with (run_root / "config.generated.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        print(f"\n=== SAME-OPERATOR RETENTION / seed_{seed} ===")
        task_rows: List[Dict[str, Any]] = []
        if args.quiet:
            console_path = run_root / "logs" / "console_capture.log"
            console_path.parent.mkdir(parents=True, exist_ok=True)
            with console_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
                system = SelfEvolveSystem(cfg)
                task_rows.extend(run_probe_phase(system, phase="pre", seed=seed, task_id_offset=1000))
                task_rows.extend(run_stress_phase(system, seed=seed))
                task_rows.extend(run_probe_phase(system, phase="post", seed=seed, task_id_offset=2000))
                save_system_artifacts(system, run_root / "results")
            print(f"Captured verbose log: {console_path}")
        else:
            system = SelfEvolveSystem(cfg)
            task_rows.extend(run_probe_phase(system, phase="pre", seed=seed, task_id_offset=1000))
            task_rows.extend(run_stress_phase(system, seed=seed))
            task_rows.extend(run_probe_phase(system, phase="post", seed=seed, task_id_offset=2000))
            save_system_artifacts(system, run_root / "results")

        write_csv(run_root / "retention_tasks.csv", task_rows)

        all_task_rows.extend(task_rows)
        summary_rows.append(summarize_seed(seed, run_root, task_rows))

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "retention_tasks.csv", all_task_rows)
    write_csv(output_root / "retention_summary.csv", summary_rows)
    write_json(output_root / "retention_summary.json", summary_rows)

    print("\nSAME-OPERATOR RETENTION SUMMARY")
    for row in summary_rows:
        print(
            f"seed_{row['seed']}: pre={row['pre_success_rate']:.3f}, "
            f"post={row['post_success_rate']:.3f}, "
            f"delta={row['delta_post_minus_pre']:+.3f}, "
            f"patches={row['patches_committed']}"
        )
    print(f"CSV: {output_root / 'retention_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
