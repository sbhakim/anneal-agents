"""
Ablation Study Experiment
Corresponds to Section XII.3 evaluation protocol.
Generates data for TABLE 2: Component Contributions.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.system import SelfEvolveSystem
from src.utils.config_loader import load_config


class AblationStudy:
    """
    Ablation experiments to measure component contributions.
    Systematically disables one component at a time.
    """

    def __init__(self, base_config_path: str = "config.yaml"):
        self.base_config = load_config(base_config_path)
        self.results_dir = Path("experiments/results/ablations")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.num_tasks = 50
        self.num_seeds = 3  # Ablations need fewer seeds (internal comparison)
        self.seeds = [42, 123, 456]

        print("=" * 70)
        print("ABLATION STUDY EXPERIMENT")
        print("=" * 70)
        print(f"Configuration: {self.num_tasks} tasks × {self.num_seeds} seeds")
        print(f"Results directory: {self.results_dir}")
        print()

    def get_ablation_configs(self) -> Dict[str, Dict]:
        """
        Define ablation configurations.
        Each disables one key component.
        """
        configs = {
            "SelfEvolve-Full": {
                "name": "SelfEvolve (Full)",
                "description": "Complete system (control)",
                "config_overrides": {}
            },

            "No-Governance": {
                "name": "Without Governance",
                "description": "FDKA without provenance, trust, gates, canary",
                "config_overrides": {
                    "governance": {
                        "provenance": {"enable": False},
                        "trust": {"alpha": 1, "beta": 1},  # Neutral prior
                        "gates": {"tau_impact": 999.0, "tau_conf": 0.0},  # No gating
                        "canary": {"num_tests": 0}  # No canary
                    }
                }
            },

            "No-Verify": {
                "name": "Without Verify-Before-Act",
                "description": "FDKA + governance, but no precondition pre-checking",
                "config_overrides": {
                    "executor": {
                        "enable_verification": False
                    }
                }
            },

            "No-Arbitration": {
                "name": "Without Metacognitive Arbitration",
                "description": "Always use S1 pathway, no adaptive routing",
                "config_overrides": {
                    "metacognition": {
                        "tau_u": 999.0,  # Never escalate to S2
                        "tau_p": 999.0,
                        "enable_reflection": False
                    }
                }
            },

            "No-FDKA": {
                "name": "Without FDKA",
                "description": "Verify + Arbitration but no self-evolution",
                "config_overrides": {
                    "fdka": {
                        "threshold": 999.0  # Effectively disable
                    }
                }
            }
        }

        return configs

    def run_single_ablation(self, ablation_name: str, config: Dict, seed: int) -> Dict:
        """Run one ablation configuration."""
        print(f"\n{'─' * 70}")
        print(f"Running: {ablation_name} (seed={seed})")
        print(f"{'─' * 70}")

        experiment_config = copy.deepcopy(self.base_config)

        # Apply ablation-specific overrides
        for key, value in config.get("config_overrides", {}).items():
            if isinstance(value, dict):
                experiment_config[key].update(value)
            else:
                experiment_config[key] = value

        experiment_config['scenario']['num_tasks'] = self.num_tasks
        experiment_config['scenario']['seed'] = seed
        experiment_config['logging']['level'] = 'WARNING'

        start_time = time.time()

        try:
            system = SelfEvolveSystem(experiment_config)
            metrics = system.run_evaluation()

            elapsed = time.time() - start_time
            summary = metrics.get_summary()

            result = {
                "ablation": ablation_name,
                "seed": seed,
                "elapsed_time": elapsed,
                "metrics": summary,
                "config": config,
                "timestamp": time.time()
            }

            print(f"\n✓ Completed in {elapsed:.1f}s")
            print(f"  Success Rate: {summary['success_rate']:.1%}")
            print(f"  TTA: {self._format_tta(summary['time_to_adapt'])}")
            print(f"  Patches: {summary['patches_accepted']}/{summary['patches_proposed']}")
            print(f"  Rollbacks: {summary.get('rollback_frequency', 0):.1f}")

            return result

        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "ablation": ablation_name,
                "seed": seed,
                "error": str(e),
                "timestamp": time.time()
            }

    def run_all_ablations(self) -> Dict[str, List[Dict]]:
        """Run complete ablation matrix."""
        configs = self.get_ablation_configs()
        all_results = {}

        total_runs = len(configs) * len(self.seeds)
        current_run = 0

        print(f"\nStarting {total_runs} ablation runs...")
        print(f"Ablations: {list(configs.keys())}")
        print(f"Seeds: {self.seeds}\n")

        for ablation_name, config in configs.items():
            ablation_results = []

            for seed in self.seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {ablation_name} × seed={seed}")

                result = self.run_single_ablation(ablation_name, config, seed)
                ablation_results.append(result)

                # Checkpoint save
                self._save_single_result(result, ablation_name, seed)

            all_results[ablation_name] = ablation_results
            self._save_ablation_results(ablation_name, ablation_results)

        self._save_all_results(all_results)
        self._export_to_csv(all_results)

        return all_results

    def _save_single_result(self, result: Dict, ablation: str, seed: int):
        """Save individual run."""
        filepath = self.results_dir / f"{ablation}_seed{seed}.json"
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

    def _save_ablation_results(self, ablation: str, results: List[Dict]):
        """Save aggregated results for one ablation."""
        filepath = self.results_dir / f"{ablation}_aggregated.json"

        aggregated = {
            "ablation": ablation,
            "num_seeds": len(results),
            "mean_metrics": self._compute_mean_metrics(results),
            "std_metrics": self._compute_std_metrics(results),
            "individual_runs": results
        }

        with open(filepath, 'w') as f:
            json.dump(aggregated, f, indent=2)

    def _save_all_results(self, all_results: Dict):
        """Save complete ablation matrix."""
        filepath = self.results_dir / "complete_ablations.json"

        summary = {
            "experiment": "ablation_study",
            "num_tasks": self.num_tasks,
            "num_seeds": self.num_seeds,
            "results": all_results,
            "timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Saved complete ablations: {filepath}")

    def _export_to_csv(self, all_results: Dict):
        """Export to CSV for Table 2."""
        import csv
        import numpy as np

        csv_path = self.results_dir / "table2_data.csv"

        # Compute deltas relative to Full system
        full_results = all_results.get("SelfEvolve-Full", [])
        full_tta = self._compute_mean_tta(full_results)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                "Configuration", "Success_Rate_Mean", "Success_Rate_Std",
                "TTA_Mean", "TTA_Std", "Delta_TTA_vs_Full",
                "Patches_Accepted", "Rollbacks", "Component_Removed"
            ])

            for ablation_name, results in all_results.items():
                valid_results = [r for r in results if "error" not in r]
                if not valid_results:
                    continue

                success_rates = [r["metrics"]["success_rate"] for r in valid_results]
                tta_mean = self._compute_mean_tta(valid_results)
                tta_std = self._compute_std_tta(valid_results)
                patches = np.mean([r["metrics"]["patches_accepted"] for r in valid_results])
                rollbacks = np.mean([r["metrics"].get("rollback_frequency", 0) for r in valid_results])

                delta_tta = tta_mean - full_tta if full_tta != float('inf') and tta_mean != float('inf') else None

                component_removed = ablation_name.replace("No-", "").replace("SelfEvolve-Full", "None")

                writer.writerow([
                    ablation_name,
                    f"{np.mean(success_rates):.3f}",
                    f"{np.std(success_rates):.3f}",
                    f"{tta_mean:.1f}" if tta_mean != float('inf') else "∞",
                    f"{tta_std:.1f}" if tta_std != float('inf') else "N/A",
                    f"{delta_tta:+.1f}" if delta_tta is not None else "N/A",
                    f"{patches:.1f}",
                    f"{rollbacks:.1f}",
                    component_removed
                ])

        print(f"💾 Exported CSV: {csv_path}")

    def _compute_mean_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean metrics."""
        import numpy as np

        valid = [r for r in results if "error" not in r]
        if not valid:
            return {}

        return {
            "success_rate": float(np.mean([r["metrics"]["success_rate"] for r in valid])),
            "tta_mean": self._compute_mean_tta(valid),
            "patches_accepted": float(np.mean([r["metrics"]["patches_accepted"] for r in valid])),
        }

    def _compute_std_metrics(self, results: List[Dict]) -> Dict:
        """Compute std metrics."""
        import numpy as np

        valid = [r for r in results if "error" not in r]
        if not valid:
            return {}

        return {
            "success_rate": float(np.std([r["metrics"]["success_rate"] for r in valid])),
        }

    def _compute_mean_tta(self, results: List[Dict]) -> float:
        """Compute mean TTA across results."""
        import numpy as np

        tta_values = []
        for r in results:
            if "error" not in r:
                for v in r["metrics"]["time_to_adapt"].values():
                    if v is not None:
                        tta_values.append(v)

        return float(np.mean(tta_values)) if tta_values else float('inf')

    def _compute_std_tta(self, results: List[Dict]) -> float:
        """Compute std of TTA."""
        import numpy as np

        tta_values = []
        for r in results:
            if "error" not in r:
                for v in r["metrics"]["time_to_adapt"].values():
                    if v is not None:
                        tta_values.append(v)

        return float(np.std(tta_values)) if len(tta_values) > 1 else float('inf')

    def _format_tta(self, tta_dict: Dict) -> str:
        """Format TTA for printing."""
        if not tta_dict:
            return "N/A"
        values = [f"{k}={v}" if v else f"{k}=∞" for k, v in tta_dict.items()]
        return ", ".join(values)

    def print_summary(self, all_results: Dict):
        """Print ablation summary."""
        import numpy as np

        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        full_results = all_results.get("SelfEvolve-Full", [])
        full_tta = self._compute_mean_tta([r for r in full_results if "error" not in r])

        print(f"\nBaseline (Full System): TTA = {full_tta:.1f} tasks")
        print(f"\nComponent Contributions (Δ TTA):")

        for ablation_name, results in all_results.items():
            if ablation_name == "SelfEvolve-Full":
                continue

            valid = [r for r in results if "error" not in r]
            if not valid:
                continue

            tta = self._compute_mean_tta(valid)
            delta = tta - full_tta if tta != float('inf') and full_tta != float('inf') else None

            component = ablation_name.replace("No-", "")
            print(f"  {component:20s}: {delta:+.1f} tasks" if delta else f"  {component:20s}: N/A")

        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {self.results_dir}")
        print("=" * 70)


def main():
    """Main entry point."""
    experiment = AblationStudy()
    all_results = experiment.run_all_ablations()
    experiment.print_summary(all_results)

    print("\n✓ Ablation study complete!")
    print(f"   Check {experiment.results_dir}/table2_data.csv for table data")


if __name__ == "__main__":
    main()