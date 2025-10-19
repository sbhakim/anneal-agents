"""
Baseline Comparison Experiment
Corresponds to Section XII.2 evaluation protocol.
Generates data for TABLE 1: Main Results vs Baselines.

UPDATES (stability + fidelity):
- Deep copy + deep-merge config overrides (prevents nested key loss and cross-run leakage).
- Replace legacy "fdka.threshold=999" kill-switch with explicit "fdka.enabled: false".
- Normalize any legacy overrides that still try to use the threshold hack.
- Force difficulty='hard' to match manuscript stress setting.
- Explicitly set FDKA enabled=True for SelfEvolve-Full to avoid leakage from previous runs.
- Add a short provider/model sanity print per run.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# FIXED: Correct import paths (no "..")
from src.core.system import SelfEvolveSystem
from src.utils.config_loader import load_config
from src.baselines.static_ns import StaticNSAgent
from src.baselines.llm_reflect import LLMReflectAgent
from src.baselines.verify_only import VerifyOnlyAgent

# Check for numpy (needed for statistics)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found. Statistical computations will use fallback.")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge 'override' into 'base' without mutating inputs.
    Values from 'override' replace or merge into 'base' recursively.
    """
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def normalize_overrides(ov: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config overrides:
    - Convert legacy 'fdka.threshold >= 900' to explicit 'fdka.enabled: false'
    - Remove contradictory settings ('enabled: false' with a small threshold)
    """
    out = copy.deepcopy(ov or {})
    fdka = out.get("fdka", {})
    if isinstance(fdka, dict):
        # Legacy kill-switch → boolean gate
        th = fdka.get("threshold", None)
        if isinstance(th, (int, float)) and th >= 900:
            fdka.pop("threshold", None)
            fdka["enabled"] = False
        # If explicitly disabled, remove tiny threshold that implies accept gating
        if fdka.get("enabled") is False and isinstance(fdka.get("threshold"), (int, float)) and fdka["threshold"] < 10:
            fdka.pop("threshold", None)
        out["fdka"] = fdka
    return out


class BaselineComparison:
    """
    Orchestrates baseline comparison experiments.
    Runs each system configuration across multiple seeds and aggregates results.
    """

    def __init__(self, base_config_path: str = "config.yaml"):
        # FIXED: Resolve config path relative to project root
        config_path = Path(__file__).parent.parent / base_config_path
        self.base_config = load_config(str(config_path))

        self.results_dir = Path("experiments/results/baseline_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Experiment parameters from manuscript Section XII
        self.num_tasks = 50  # Standard evaluation length
        self.num_seeds = 5   # Statistical significance (5 runs)
        self.seeds = [42, 123, 456, 789, 1337]

        print("=" * 70)
        print("BASELINE COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"Configuration: {self.num_tasks} tasks × {self.num_seeds} seeds")
        print(f"Results directory: {self.results_dir}")
        print()

    def get_system_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define configurations for each system variant.
        Maps to Section XII.2 baseline definitions.
        """
        configs = {
            "SelfEvolve-Full": {
                "name": "SelfEvolve (Full)",
                "description": "Complete system with FDKA, metacognition, governance",
                "config_overrides": {
                    # All features enabled (baseline from config.yaml). We still force-enable below per run.
                }
            },

            "Static-NS": {
                "name": "Static Neurosymbolic",
                "description": "No FDKA, no learning, fixed operators",
                "config_overrides": {
                    # ✅ Modern, explicit disable
                    "fdka": {"enabled": False},
                    "metacognition": {"enable_reflection": False},
                    "governance": {"provenance": {"enable": False}},
                }
            },

            "LLM-Reflect": {
                "name": "LLM + Textual Reflection",
                "description": "ReAct/Reflexion-style with text memory, no symbolic edits",
                "config_overrides": {
                    "fdka": {"enabled": False},  # Disable symbolic patches
                    "metacognition": {"enable_reflection": True},
                    "executor": {"enable_verification": False},  # LLM handles reasoning
                }
            },

            "Verify-Only": {
                "name": "Verify-Before-Act Only",
                "description": "Precondition checking without self-evolution",
                "config_overrides": {
                    "fdka": {"enabled": False},  # No FDKA
                    "metacognition": {"enable_reflection": False},
                    "executor": {"enable_verification": True},
                }
            }
        }

        return configs

    def run_single_experiment(self, system_name: str, config: Dict, seed: int) -> Dict:
        """
        Run a single experiment: one system × one seed.

        Returns:
            Dictionary with metrics and metadata
        """
        print(f"\n{'─' * 70}")
        print(f"Running: {system_name} (seed={seed})")
        print(f"{'─' * 70}")

        # Deep copy base config to avoid cross-run mutation
        experiment_config = copy.deepcopy(self.base_config)

        # Apply normalized system-specific overrides via deep merge
        overrides = normalize_overrides(config.get("config_overrides", {}))
        experiment_config = deep_merge(experiment_config, overrides)

        # Force scenario parameters (stress setting to match manuscript)
        experiment_config.setdefault('scenario', {})
        experiment_config['scenario']['num_tasks'] = self.num_tasks
        experiment_config['scenario']['failure_injector_seed'] = seed
        experiment_config['scenario']['difficulty'] = 'hard'  # manuscript stress setting

        # Disable verbose logging for batch runs
        experiment_config.setdefault('logging', {})
        experiment_config['logging']['level'] = 'WARNING'

        # Ensure the Full system explicitly enables FDKA (defensive against leakage)
        experiment_config.setdefault('fdka', {})
        if system_name == "SelfEvolve-Full":
            experiment_config['fdka']['enabled'] = True

        # Short provider/model sanity print
        pe = experiment_config.get('fdka', {}).get('propose_edit', {})
        print(f"CFG → FDKA.enabled={experiment_config.get('fdka', {}).get('enabled', True)} | "
              f"provider={pe.get('llm_provider')} | model={pe.get('model')}")

        start_time = time.time()

        try:
            # Initialize system/agent
            if system_name == "SelfEvolve-Full":
                system = SelfEvolveSystem(experiment_config)
                metrics = system.run_evaluation()
            elif system_name == "Static-NS":
                agent = StaticNSAgent(experiment_config)
                metrics = agent.run_evaluation()
            elif system_name == "LLM-Reflect":
                agent = LLMReflectAgent(experiment_config)
                metrics = agent.run_evaluation()
            elif system_name == "Verify-Only":
                agent = VerifyOnlyAgent(experiment_config)
                metrics = agent.run_evaluation()
            else:
                raise ValueError(f"Unknown system: {system_name}")

            elapsed = time.time() - start_time

            # FIXED: Better error handling for metrics extraction
            if hasattr(metrics, 'get_summary'):
                summary = metrics.get_summary()
            elif isinstance(metrics, dict):
                summary = metrics
            else:
                raise ValueError(f"Unexpected metrics type: {type(metrics)}")

            result = {
                "system": system_name,
                "seed": seed,
                "num_tasks": self.num_tasks,
                "elapsed_time": elapsed,
                "metrics": summary,
                "config": config,
                "timestamp": time.time()
            }

            # Print summary
            print(f"\n✓ Completed in {elapsed:.1f}s")
            print(f"  Success Rate: {summary['success_rate']:.1%}")
            print(f"  RFR: {summary['repeat_failure_rate']:.1%}")
            print(f"  TTA: {self._format_tta(summary['time_to_adapt'])}")

            return result

        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "system": system_name,
                "seed": seed,
                "error": str(e),
                "timestamp": time.time()
            }

    def run_all_experiments(self) -> Dict[str, List[Dict]]:
        """
        Run complete baseline comparison matrix.

        Returns:
            Dictionary mapping system_name -> list of results (one per seed)
        """
        configs = self.get_system_configs()
        all_results = {}

        total_runs = len(configs) * len(self.seeds)
        current_run = 0

        print(f"\nStarting {total_runs} experiment runs...")
        print(f"Systems: {list(configs.keys())}")
        print(f"Seeds: {self.seeds}\n")

        for system_name, config in configs.items():
            system_results = []

            for seed in self.seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {system_name} × seed={seed}")

                result = self.run_single_experiment(system_name, config, seed)
                system_results.append(result)

                # Save individual result immediately (checkpoint)
                self._save_single_result(result, system_name, seed)

            all_results[system_name] = system_results

            # Save aggregated results per system
            self._save_system_results(system_name, system_results)

        # Save complete results matrix
        self._save_all_results(all_results)

        return all_results

    def _save_single_result(self, result: Dict, system: str, seed: int):
        """Save individual run result."""
        filepath = self.results_dir / f"{system}_seed{seed}.json"
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

    def _save_system_results(self, system: str, results: List[Dict]):
        """Save aggregated results for one system across all seeds."""
        filepath = self.results_dir / f"{system}_aggregated.json"

        # Compute statistics across seeds
        aggregated = {
            "system": system,
            "num_seeds": len(results),
            "seeds": [r["seed"] for r in results if "error" not in r],
            "mean_metrics": self._compute_mean_metrics(results),
            "std_metrics": self._compute_std_metrics(results),
            "individual_runs": results
        }

        with open(filepath, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"\n💾 Saved aggregated results: {filepath}")

    def _save_all_results(self, all_results: Dict[str, List[Dict]]):
        """Save complete results matrix."""
        filepath = self.results_dir / "complete_results.json"

        summary = {
            "experiment": "baseline_comparison",
            "num_tasks": self.num_tasks,
            "num_seeds": self.num_seeds,
            "seeds": self.seeds,
            "systems": list(all_results.keys()),
            "results": all_results,
            "timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Saved complete results: {filepath}")

        # Also save CSV for easy table generation
        self._export_to_csv(all_results)

    def _export_to_csv(self, all_results: Dict[str, List[Dict]]):
        """Export results to CSV for easy table generation."""
        import csv

        csv_path = self.results_dir / "table1_data.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "System", "Seed", "Success_Rate", "RFR", "CSR",
                "TTA_Mean", "RF", "RP", "HI", "Patches_Proposed", "Patches_Accepted"
            ])

            # Data rows
            for system_name, results in all_results.items():
                for result in results:
                    if "error" in result:
                        continue

                    metrics = result["metrics"]
                    tta_vals = [v for v in metrics.get("time_to_adapt", {}).values() if v is not None]
                    tta_mean = (sum(tta_vals) / len(tta_vals)) if tta_vals else float('inf')

                    writer.writerow([
                        system_name,
                        result["seed"],
                        f"{metrics.get('success_rate', 0.0):.3f}",
                        f"{metrics.get('repeat_failure_rate', 0.0):.3f}",
                        f"{metrics.get('constraint_satisfaction_rate', 0.0):.3f}",
                        f"{tta_mean:.1f}" if tta_mean != float('inf') else "∞",
                        f"{metrics.get('rollback_frequency', 0.0):.1f}",
                        f"{metrics.get('rollback_precision', 0.0):.3f}",
                        metrics.get('human_interventions', 0),
                        metrics.get('patches_proposed', 0),
                        metrics.get('patches_accepted', 0)
                    ])

        print(f"💾 Exported CSV: {csv_path}")

    def _compute_mean_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean across seeds."""
        keys = ["success_rate", "repeat_failure_rate", "constraint_satisfaction_rate"]
        means = {}

        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {k: 0.0 for k in keys}

        for key in keys:
            values = [r["metrics"].get(key, 0.0) for r in valid_results]
            if HAS_NUMPY:
                means[key] = float(np.mean(values))
            else:
                means[key] = sum(values) / len(values) if values else 0.0

        # TTA: mean of non-None values
        tta_values = []
        for r in valid_results:
            for v in r["metrics"].get("time_to_adapt", {}).values():
                if v is not None:
                    tta_values.append(v)

        if HAS_NUMPY:
            means["tta_mean"] = float(np.mean(tta_values)) if tta_values else None
        else:
            means["tta_mean"] = sum(tta_values) / len(tta_values) if tta_values else None

        return means

    def _compute_std_metrics(self, results: List[Dict]) -> Dict:
        """Compute standard deviation across seeds."""
        keys = ["success_rate", "repeat_failure_rate", "constraint_satisfaction_rate"]
        stds = {}

        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {k: 0.0 for k in keys}

        for key in keys:
            values = [r["metrics"].get(key, 0.0) for r in valid_results]
            if HAS_NUMPY:
                stds[key] = float(np.std(values))
            else:
                # Simple std calculation without numpy
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    stds[key] = variance ** 0.5
                else:
                    stds[key] = 0.0

        return stds

    def _format_tta(self, tta_dict: Dict) -> str:
        """Format TTA dictionary for printing."""
        if not tta_dict:
            return "N/A"
        values = [f"{k}={v}" if v is not None else f"{k}=∞" for k, v in tta_dict.items()]
        return ", ".join(values)

    def print_summary(self, all_results: Dict[str, List[Dict]]):
        """Print experiment summary."""
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)

        for system_name, results in all_results.items():
            print(f"\n{system_name}:")

            valid_results = [r for r in results if "error" not in r]
            if not valid_results:
                print("  ✗ All runs failed")
                continue

            # Compute statistics
            success_rates = [r["metrics"].get("success_rate", 0.0) for r in valid_results]
            rfr_rates = [r["metrics"].get("repeat_failure_rate", 0.0) for r in valid_results]

            if HAS_NUMPY:
                sr_mean, sr_std = np.mean(success_rates), np.std(success_rates)
                rfr_mean, rfr_std = np.mean(rfr_rates), np.std(rfr_rates)
            else:
                sr_mean = sum(success_rates) / len(success_rates)
                sr_std = 0.0 if len(success_rates) == 1 else (
                    sum((x - sr_mean) ** 2 for x in success_rates) / len(success_rates)
                ) ** 0.5
                rfr_mean = sum(rfr_rates) / len(rfr_rates)
                rfr_std = 0.0

            print(f"  Success Rate: {sr_mean:.1%} ± {sr_std:.1%}")
            print(f"  RFR: {rfr_mean:.1%} ± {rfr_std:.1%}")
            print(f"  Completed: {len(valid_results)}/{len(results)} runs")

        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {self.results_dir}")
        print("=" * 70)


def main():
    """Main entry point."""
    experiment = BaselineComparison()
    all_results = experiment.run_all_experiments()
    experiment.print_summary(all_results)

    print("\n✓ Baseline comparison complete!")
    print(f"   Check {experiment.results_dir}/table1_data.csv for table data")


if __name__ == "__main__":
    main()
