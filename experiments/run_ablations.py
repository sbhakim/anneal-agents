"""
Ablation Study Experiment
Corresponds to Section XII.3 evaluation protocol.
Generates data for TABLE 2: Component Contributions.

UPDATED:
- Fixed config path resolution (relative to project root)
- Enhanced debugging and error handling
- Improved progress tracking and statistics
- Added validation for ablation configurations
- CRITICAL FIX: Integrated difficulty levels to enable component stress testing
- ADDED: print_summary method for evaluation orchestrator compatibility
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.system import SelfEvolveSystem
from src.utils.config_loader import load_config

# Check for numpy availability
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found. Using fallback statistics.")


class AblationStudy:
    """
    Ablation experiments to measure component contributions.
    Systematically disables one component at a time.
    """

    def __init__(self, base_config_path: str = "config.yaml"):
        """
        Initialize ablation study.

        Args:
            base_config_path: Path to base configuration file (relative to project root)
        """
        config_path = Path(__file__).parent.parent / base_config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.base_config = load_config(str(config_path))

        self.results_dir = Path("experiments/results/ablations")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.num_seeds = 3  # Ablations need fewer seeds (internal comparison)
        self.seeds = [42, 123, 456]

        # ✅ Define difficulty levels for the stress test
        self.difficulty_levels = ['easy', 'normal', 'hard']
        self.task_counts = {'easy': 30, 'normal': 50, 'hard': 20}

        print("=" * 70)
        print("ABLATION STUDY EXPERIMENT (COMPONENT STRESS TEST)")
        print("=" * 70)
        print(f"Configuration: {len(self.difficulty_levels)} difficulties × {self.num_seeds} seeds")
        print(f"Results directory: {self.results_dir}")
        print(f"Base config loaded from: {config_path}")
        print()

    def get_ablation_configs(self) -> Dict[str, Dict]:
        """
        Define ablation configurations.
        Each disables one key component to measure its contribution.

        Returns:
            Dictionary mapping ablation name to configuration
        """
        configs = {
            "SelfEvolve-Full": {
                "name": "SelfEvolve (Full)",
                "description": "Complete system (control baseline)",
                "config_overrides": {}
            },

            "No-Governance": {
                "name": "Without Governance",
                "description": "FDKA without provenance, trust, gates, canary",
                "config_overrides": {
                    "governance": {
                        "provenance": {"enable": False},
                        "trust": {"alpha": 1, "beta": 1},  # Neutral prior (no learning)
                        "gates": {"tau_impact": 999.0, "tau_conf": 0.0},  # Disable gating
                        "canary": {"num_tests": 0}  # Disable canary testing
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
                        "tau_p": 999.0,  # Never verify
                        "enable_reflection": False
                    }
                }
            },

            "No-FDKA": {
                "name": "Without FDKA",
                "description": "Verify + Arbitration but no self-evolution (static operators)",
                "config_overrides": {
                    "fdka": {
                        "threshold": 999.0  # Effectively disable (no patches pass)
                    }
                }
            }
        }

        return configs

    def run_single_ablation(self, ablation_name: str, config: Dict, seed: int, difficulty: str) -> Dict:
        """
        Run one ablation configuration with a specific seed and difficulty.

        Args:
            ablation_name: Name of the ablation variant
            config: Ablation configuration with overrides
            seed: Random seed for reproducibility
            difficulty: The scenario difficulty ('easy', 'normal', 'hard')

        Returns:
            Dictionary with results and metrics
        """
        print(f"\n{'─' * 70}")
        print(f"Running: {ablation_name} (seed={seed}, difficulty={difficulty})")
        print(f"{'─' * 70}")

        experiment_config = copy.deepcopy(self.base_config)

        for key, value in config.get("config_overrides", {}).items():
            if isinstance(value, dict) and key in experiment_config:
                experiment_config[key].update(value)
            else:
                experiment_config[key] = value

        # ✅ Set experiment parameters based on difficulty
        experiment_config['scenario']['difficulty'] = difficulty
        experiment_config['scenario']['num_tasks'] = self.task_counts[difficulty]
        experiment_config['scenario']['failure_injector_seed'] = seed
        experiment_config['logging']['level'] = 'WARNING'

        start_time = time.time()

        try:
            system = SelfEvolveSystem(experiment_config)
            metrics = system.run_evaluation()
            elapsed = time.time() - start_time

            if hasattr(metrics, 'get_summary'):
                summary = metrics.get_summary()
            elif isinstance(metrics, dict):
                summary = metrics
            else:
                raise ValueError(f"Unexpected metrics type: {type(metrics)}")

            result = {
                "ablation": ablation_name,
                "seed": seed,
                "difficulty": difficulty,
                "elapsed_time": elapsed,
                "metrics": summary,
                "config": config,
                "timestamp": time.time()
            }

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
                "ablation": ablation_name,
                "seed": seed,
                "difficulty": difficulty,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }

    def run_all_ablations(self) -> Dict[str, List[Dict]]:
        """
        Run complete ablation matrix across all difficulties and seeds.
        """
        configs = self.get_ablation_configs()
        all_results = {}

        total_runs = len(configs) * len(self.seeds) * len(self.difficulty_levels)
        current_run = 0

        print(f"\n{'=' * 70}")
        print(f"STARTING ABLATION MATRIX")
        print(f"{'=' * 70}")
        print(f"Total configurations: {len(configs)}")
        print(f"Seeds per config: {len(self.seeds)}")
        print(f"Difficulty levels: {self.difficulty_levels}")
        print(f"Total runs: {total_runs}")
        print(f"\nAblations to run:")
        for i, (name, cfg) in enumerate(configs.items(), 1):
            print(f"  {i}. {name}: {cfg['description']}")
        print(f"\nSeeds: {self.seeds}")
        print(f"{'=' * 70}\n")

        start_time = time.time()

        for ablation_idx, (ablation_name, config) in enumerate(configs.items(), 1):
            ablation_results = []

            # ✅ Loop through difficulty levels
            for difficulty_idx, difficulty in enumerate(self.difficulty_levels, 1):
                for seed_idx, seed in enumerate(self.seeds, 1):
                    current_run += 1
                    print(f"\n[Run {current_run}/{total_runs}] {ablation_name} × seed={seed} × difficulty={difficulty}")

                    try:
                        result = self.run_single_ablation(ablation_name, config, seed, difficulty)
                        ablation_results.append(result)
                        self._save_single_result(result, ablation_name, seed, difficulty)
                    except KeyboardInterrupt:
                        print(f"\n⚠️  Ablation study interrupted by user")
                        raise
                    except Exception as e:
                        print(f"  ❌ EXCEPTION during run: {e}")
                        error_result = {
                            "ablation": ablation_name, "seed": seed, "difficulty": difficulty,
                            "error": str(e), "timestamp": time.time()
                        }
                        ablation_results.append(error_result)
                        self._save_single_result(error_result, ablation_name, seed, difficulty)

            all_results[ablation_name] = ablation_results
            self._save_ablation_results(ablation_name, ablation_results)

        elapsed_total = time.time() - start_time

        print(f"\n{'=' * 70}")
        print(f"ABLATION MATRIX COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total runs executed: {total_runs}")
        successful_runs = sum(1 for results in all_results.values()
                            for r in results if "error" not in r)
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"  Success rate: {successful_runs / total_runs * 100:.1f}%")
        print(f"Total elapsed time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
        print(f"Average time per run: {elapsed_total/total_runs:.1f}s")
        print(f"{'=' * 70}\n")

        self._save_all_results(all_results)
        self._export_to_csv(all_results)

        return all_results

    def print_summary(self, all_results: Dict[str, List[Dict]]):
        """
        Print formatted summary of ablation study results.

        ✅ ADDED: This method is called by run_full_evaluation.py orchestrator.

        Args:
            all_results: Dictionary mapping ablation names to their result lists
        """
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        # Extract baseline (full system) metrics
        full_system_results = all_results.get('SelfEvolve-Full', [])

        if not full_system_results:
            print("⚠️  No baseline results found")
            return

        # Get valid baseline runs (no errors)
        valid_baseline = [r for r in full_system_results if "error" not in r]

        if not valid_baseline:
            print("⚠️  No valid baseline runs")
            return

        # Compute baseline averages across all difficulties
        baseline_sr = self._safe_mean([r['metrics']['success_rate'] for r in valid_baseline])
        baseline_rfr = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_baseline])
        baseline_tta = self._compute_mean_tta(valid_baseline)

        print(f"\nBaseline (Full System):")
        print(f"  Success Rate: {baseline_sr:.1%}")
        print(f"  RFR: {baseline_rfr:.1%}")
        print(f"  TTA: {baseline_tta:.1f} tasks")

        print(f"\nComponent Contributions (Δ vs Full System):")
        print("-" * 70)
        print(f"{'Component':<25} {'Δ Success Rate':>15} {'Δ RFR':>10} {'Δ TTA':>10} {'Status':>8}")
        print("-" * 70)

        for ablation_name, results in all_results.items():
            if ablation_name == 'SelfEvolve-Full':
                continue  # Skip baseline

            component = ablation_name.replace('No-', '')
            valid_runs = [r for r in results if "error" not in r]

            if not valid_runs:
                print(f"{component:<25} {'N/A':>15} {'N/A':>10} {'N/A':>10} {'❌':>8}")
                continue

            sr = self._safe_mean([r['metrics']['success_rate'] for r in valid_runs])
            rfr = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_runs])
            tta = self._compute_mean_tta(valid_runs)

            delta_sr = (sr - baseline_sr) * 100  # percentage points
            delta_rfr = (rfr - baseline_rfr) * 100
            delta_tta = tta - baseline_tta if tta != float('inf') and baseline_tta != float('inf') else float('inf')

            # Status indicator
            if abs(delta_sr) < 1 and abs(delta_rfr) < 1 and abs(delta_tta) < 2:
                status = "✅"  # No significant impact
            elif delta_sr < -5 or delta_rfr > 5 or delta_tta > 5:
                status = "❌"  # Significant degradation
            else:
                status = "⚠️"  # Moderate impact

            tta_str = f"+{delta_tta:.1f}" if delta_tta != float('inf') else "∞"

            print(f"{component:<25} {delta_sr:>+14.1f}% {delta_rfr:>+9.1f}% {tta_str:>10} {status:>8}")

        print("-" * 70)
        print("\n💡 Interpretation:")
        print("   • Negative Δ Success Rate = Component removal hurts performance")
        print("   • Positive Δ RFR = Component removal increases repeat failures")
        print("   • Positive Δ TTA = Component removal slows adaptation")
        print("   • ∞ TTA = No adaptation occurred (system cannot learn)")

        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {self.results_dir}")
        print(f"✓ CSV data for Table 2: {self.results_dir / 'table2_data.csv'}")
        print("=" * 70)

    def _save_single_result(self, result: Dict, ablation: str, seed: int, difficulty: str):
        """Save individual run result to JSON file."""
        filepath = self.results_dir / f"{ablation}_{difficulty}_seed{seed}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Failed to save result: {e}")

    def _save_ablation_results(self, ablation: str, results: List[Dict]):
        """Save aggregated results for one ablation across all seeds and difficulties."""
        filepath = self.results_dir / f"{ablation}_aggregated.json"

        aggregated = {
            "ablation": ablation,
            "num_runs": len(results),
            "runs": results,
            "timestamp": time.time()
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(aggregated, f, indent=2)
            print(f"  💾 Saved aggregated results: {filepath.name}")
        except Exception as e:
            print(f"  ⚠️  Failed to save aggregated results: {e}")

    def _save_all_results(self, all_results: Dict):
        """Save complete ablation matrix results."""
        filepath = self.results_dir / "complete_ablations.json"

        summary = {
            "experiment": "ablation_study_stress_test",
            "seeds": self.seeds,
            "difficulties": self.difficulty_levels,
            "task_counts": self.task_counts,
            "num_configurations": len(all_results),
            "total_runs": sum(len(results) for results in all_results.values()),
            "results": all_results,
            "timestamp": time.time()
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Saved complete ablations: {filepath}")
        except Exception as e:
            print(f"\n⚠️  Failed to save complete results: {e}")

    def _export_to_csv(self, all_results: Dict):
        """Export results to CSV for Table 2 in manuscript, including difficulty breakdown."""
        import csv
        csv_path = self.results_dir / "table2_data.csv"

        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Configuration", "Difficulty", "Success_Rate_Mean", "Success_Rate_Std",
                    "RFR_Mean", "RFR_Std", "TTA_Mean", "TTA_Std",
                    "Patches_Accepted", "Rollbacks", "Component_Removed"
                ])

                for ablation_name, results in all_results.items():
                    for difficulty in self.difficulty_levels:
                        valid_runs = [r for r in results
                                    if "error" not in r and r.get('difficulty') == difficulty]

                        if not valid_runs:
                            continue

                        sr_mean = self._safe_mean([r['metrics']['success_rate'] for r in valid_runs])
                        sr_std = self._safe_std([r['metrics']['success_rate'] for r in valid_runs])
                        rfr_mean = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_runs])
                        rfr_std = self._safe_std([r['metrics']['repeat_failure_rate'] for r in valid_runs])
                        tta_mean = self._compute_mean_tta(valid_runs)
                        tta_std = self._compute_std_tta(valid_runs)

                        patches = self._safe_mean([
                            r['metrics'].get('patches_accepted', 0) for r in valid_runs
                        ])
                        rollbacks = self._safe_mean([
                            r['metrics'].get('rollback_frequency', 0) for r in valid_runs
                        ])

                        component = ablation_name.replace("No-", "").replace("SelfEvolve-Full", "None")

                        writer.writerow([
                            ablation_name,
                            difficulty,
                            f"{sr_mean:.3f}",
                            f"{sr_std:.3f}",
                            f"{rfr_mean:.3f}",
                            f"{rfr_std:.3f}",
                            f"{tta_mean:.1f}" if tta_mean != float('inf') else "∞",
                            f"{tta_std:.1f}" if tta_std != float('inf') else "∞",
                            f"{patches:.1f}",
                            f"{rollbacks:.1f}",
                            component
                        ])

            print(f"💾 Exported CSV: {csv_path}")

        except Exception as e:
            print(f"\n⚠️  Failed to export CSV: {e}")
            import traceback
            traceback.print_exc()

    def _compute_mean_tta(self, results: List[Dict]) -> float:
        """Compute mean Time-to-Adapt across results."""
        tta_values = []
        for r in results:
            if "error" not in r:
                for v in r["metrics"]["time_to_adapt"].values():
                    if v is not None:
                        tta_values.append(v)
        return self._safe_mean(tta_values) if tta_values else float('inf')

    def _compute_std_tta(self, results: List[Dict]) -> float:
        """Compute standard deviation of Time-to-Adapt across results."""
        tta_values = []
        for r in results:
            if "error" not in r:
                for v in r["metrics"]["time_to_adapt"].values():
                    if v is not None:
                        tta_values.append(v)
        return self._safe_std(tta_values) if len(tta_values) >= 2 else float('inf')

    def _safe_mean(self, values: List[float]) -> float:
        """Compute mean, handling empty lists."""
        if not values:
            return 0.0
        if HAS_NUMPY:
            return float(np.mean(values))
        return sum(values) / len(values)

    def _safe_std(self, values: List[float]) -> float:
        """Compute standard deviation, handling small samples."""
        if not values or len(values) < 2:
            return 0.0
        if HAS_NUMPY:
            return float(np.std(values))
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _format_tta(self, tta_dict: Dict) -> str:
        """Format TTA dictionary for display."""
        if not tta_dict:
            return "N/A"
        values = [f"{k}={v}" if v is not None else f"{k}=∞"
                 for k, v in tta_dict.items()]
        return ", ".join(values) if values else "N/A"


def main():
    """Main entry point for ablation study."""
    try:
        experiment = AblationStudy()
        all_results = experiment.run_all_ablations()

        # ✅ Print summary
        experiment.print_summary(all_results)

        print("\n✓ Ablation study complete!")
        print(f"   Check {experiment.results_dir}/table2_data.csv for table data")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())