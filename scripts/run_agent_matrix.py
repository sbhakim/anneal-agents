#!/usr/bin/env python3
"""Run a reproducible agent/scenario matrix from a single base config."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml


DEFAULT_AGENTS = ["anneal", "react", "reflexion"]
DEFAULT_SCENARIOS = ["travel_planning", "ecommerce", "travel_stochastic", "itsm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ANNEAL agent/scenario matrix.")
    parser.add_argument("--base-config", default="config.yaml", help="Base YAML config to clone.")
    parser.add_argument("--mode", choices=["demo", "eval"], default="eval")
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output-root", default="/tmp/anneal_openai_matrix")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--conda-env", default="hysym")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7],
        help="Task-generation seeds to sweep. Failure-injector seeds are derived deterministically.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def scenario_overrides(scenario: str) -> dict:
    if scenario == "travel_planning":
        return {"name": "travel_planning", "difficulty": "hard"}
    if scenario == "ecommerce":
        return {"name": "ecommerce", "difficulty": "easy"}
    if scenario == "travel_stochastic":
        return {
            "name": "travel_stochastic",
            "difficulty": "hard",
            "transient_failure_rate": 0.15,
            "uncertainty_injection_rate": 0.20,
            "policy_shift_rate": 0.10,
            "resource_contention_rate": 0.05,
        }
    if scenario == "itsm":
        return {"name": "itsm", "difficulty": "hard"}
    raise ValueError(f"Unsupported scenario: {scenario}")


def build_run_config(
    base: dict,
    agent: str,
    scenario: str,
    provider: str,
    model: str,
    run_root: Path,
    seed: int,
) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("fdka", {}).setdefault("propose_edit", {})
    config.setdefault("experiment", {})
    config.setdefault("scenario", {})
    config.setdefault("output", {})
    config.setdefault("logging", {})
    config.setdefault("governance", {}).setdefault("provenance", {})

    config.setdefault("system", {})
    config["experiment"]["agent_to_run"] = agent
    config["fdka"]["propose_edit"]["llm_provider"] = provider
    config["fdka"]["propose_edit"]["model"] = model
    config["system"]["domain"] = "travel_planning" if scenario == "travel_stochastic" else scenario
    config["scenario"].update(scenario_overrides(scenario))
    config["scenario"]["task_generation_seed"] = int(seed)
    config["scenario"]["failure_injector_seed"] = int(seed) + 101
    config["output"]["results_dir"] = str(run_root / "results")
    config["output"]["plots_dir"] = str(run_root / "results" / "plots")
    config["logging"]["log_file"] = str(run_root / "logs" / f"{agent}.log")
    config["governance"]["provenance"]["log_path"] = str(run_root / "logs" / "provenance.jsonl")
    return config


def run_one(config_path: Path, workdir: Path, conda_env: str, python_bin: str, mode: str, log_path: Path) -> int:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        python_bin,
        "main.py",
        "--config",
        str(config_path),
        "--mode",
        mode,
    ]
    env = dict(os.environ)
    env["CONDA_NO_PLUGINS"] = "true"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    workdir = base_config_path.parent
    output_root = Path(args.output_root).resolve()
    base = load_yaml(base_config_path)

    failures: List[str] = []
    for agent in args.agents:
        for scenario in args.scenarios:
            for seed in args.seeds:
                run_root = output_root / agent / scenario / f"seed_{seed}"
                metrics_path = run_root / "results" / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    print(f"SKIP {agent}/{scenario}/seed_{seed}: metrics already exist at {metrics_path}")
                    continue

                config = build_run_config(
                    base,
                    agent,
                    scenario,
                    args.provider,
                    args.model,
                    run_root,
                    seed,
                )
                config_path = run_root / "config.generated.yaml"
                dump_yaml(config_path, config)

                print(f"\n=== RUN {agent} / {scenario} / seed_{seed} ===")
                code = run_one(
                    config_path=config_path,
                    workdir=workdir,
                    conda_env=args.conda_env,
                    python_bin=args.python_bin,
                    mode=args.mode,
                    log_path=run_root / "logs" / "console.log",
                )
                if code != 0:
                    failures.append(f"{agent}/{scenario}/seed_{seed} (exit={code})")

    if failures:
        print("\nFAILED RUNS:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("\nAll requested runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
