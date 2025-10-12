"""
Ablation Study Experiment
Corresponds to Section XII.3 evaluation protocol.
Generates data for TABLE 2: Component Contributions.

UPDATED:
- Fixed config path resolution (relative to project root)
- Enhanced debugging and error handling
- Improved progress tracking and statistics
- Added validation for ablation configurations
- Fixed seed parameter naming for consistency with baseline comparison
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
        # FIXED: Resolve config path relative to project root
        config_path = Path(__file__).parent.parent / base_config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.base_config = load_config(str(config_path))

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

    def run_single_ablation(self, ablation_name: str, config: Dict, seed: int) -> Dict:
        """
        Run one ablation configuration with a specific seed.

        Args:
            ablation_name: Name of the ablation variant
            config: Ablation configuration with overrides
            seed: Random seed for reproducibility

        Returns:
            Dictionary with results and metrics
        """
        print(f"\n{'─' * 70}")
        print(f"Running: {ablation_name} (seed={seed})")
        print(f"{'─' * 70}")

        # Deep copy base config
        experiment_config = copy.deepcopy(self.base_config)

        # Apply ablation-specific overrides
        for key, value in config.get("config_overrides", {}).items():
            if isinstance(value, dict) and key in experiment_config:
                # Deep merge for nested configs
                experiment_config[key].update(value)
            else:
                experiment_config[key] = value

        # Set experiment parameters
        experiment_config['scenario']['num_tasks'] = self.num_tasks

        # FIXED: Use correct seed parameter name (matches baseline_comparison.py)
        experiment_config['scenario']['failure_injector_seed'] = seed

        # Reduce logging verbosity for batch runs
        experiment_config['logging']['level'] = 'WARNING'

        start_time = time.time()

        try:
            # Initialize and run system
            system = SelfEvolveSystem(experiment_config)
            metrics = system.run_evaluation()

            elapsed = time.time() - start_time

            # Extract metrics summary
            if hasattr(metrics, 'get_summary'):
                summary = metrics.get_summary()
            elif isinstance(metrics, dict):
                summary = metrics
            else:
                raise ValueError(f"Unexpected metrics type: {type(metrics)}")

            result = {
                "ablation": ablation_name,
                "seed": seed,
                "elapsed_time": elapsed,
                "metrics": summary,
                "config": config,
                "timestamp": time.time()
            }

            # Print run summary
            print(f"\n✓ Completed in {elapsed:.1f}s")
            print(f"  Success Rate: {summary['success_rate']:.1%}")
            print(f"  RFR: {summary['repeat_failure_rate']:.1%}")
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
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }

    def run_all_ablations(self) -> Dict[str, List[Dict]]:
        """
        Run complete ablation matrix with enhanced debugging and error handling.

        Returns:
            Dictionary mapping ablation name to list of results (one per seed)
        """
        configs = self.get_ablation_configs()
        all_results = {}

        total_runs = len(configs) * len(self.seeds)
        current_run = 0

        print(f"\n{'=' * 70}")
        print(f"STARTING ABLATION MATRIX")
        print(f"{'=' * 70}")
        print(f"Total configurations: {len(configs)}")
        print(f"Seeds per config: {len(self.seeds)}")
        print(f"Total runs: {total_runs}")
        print(f"\nAblations to run:")
        for i, (name, cfg) in enumerate(configs.items(), 1):
            print(f"  {i}. {name}: {cfg['description']}")
        print(f"\nSeeds: {self.seeds}")
        print(f"{'=' * 70}\n")

        # Track successful and failed runs
        successful_runs = 0
        failed_runs = 0
        start_time = time.time()

        for ablation_idx, (ablation_name, config) in enumerate(configs.items(), 1):
            print(f"\n{'─' * 70}")
            print(f"ABLATION {ablation_idx}/{len(configs)}: {ablation_name}")
            print(f"Description: {config.get('description', 'N/A')}")
            print(f"{'─' * 70}")

            ablation_results = []

            for seed_idx, seed in enumerate(self.seeds, 1):
                current_run += 1
                print(f"\n[Run {current_run}/{total_runs}] {ablation_name} × seed={seed}")
                print(f"  (Ablation {ablation_idx}/{len(configs)}, Seed {seed_idx}/{len(self.seeds)})")

                try:
                    result = self.run_single_ablation(ablation_name, config, seed)

                    # Check if result indicates success or error
                    if "error" in result:
                        print(f"  ❌ Run failed with error: {result['error'][:100]}")
                        failed_runs += 1
                    else:
                        print(f"  ✅ Run completed successfully")
                        successful_runs += 1

                    ablation_results.append(result)

                    # Checkpoint save
                    self._save_single_result(result, ablation_name, seed)

                except KeyboardInterrupt:
                    print(f"\n⚠️  Ablation study interrupted by user")
                    print(f"   Completed: {successful_runs}/{current_run} runs")
                    print(f"   Elapsed time: {time.time() - start_time:.1f}s")
                    raise

                except Exception as e:
                    print(f"  ❌ EXCEPTION during run: {e}")
                    import traceback
                    traceback.print_exc()

                    # Record the error but continue
                    error_result = {
                        "ablation": ablation_name,
                        "seed": seed,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": time.time()
                    }
                    ablation_results.append(error_result)
                    failed_runs += 1

                    # Still save the error result for debugging
                    self._save_single_result(error_result, ablation_name, seed)

            # Save aggregated results for this ablation
            all_results[ablation_name] = ablation_results
            self._save_ablation_results(ablation_name, ablation_results)

            # Print summary for this ablation
            valid_results = [r for r in ablation_results if "error" not in r]
            print(f"\n{'─' * 70}")
            print(f"ABLATION COMPLETE: {ablation_name}")
            print(f"  Successful runs: {len(valid_results)}/{len(ablation_results)}")

            if valid_results:
                success_rates = [r["metrics"]["success_rate"] for r in valid_results]
                rfr_rates = [r["metrics"]["repeat_failure_rate"] for r in valid_results]

                if HAS_NUMPY:
                    print(f"  Mean success rate: {np.mean(success_rates):.1%} ± {np.std(success_rates):.1%}")
                    print(f"  Mean RFR: {np.mean(rfr_rates):.1%}")
                else:
                    mean_sr = sum(success_rates) / len(success_rates)
                    print(f"  Mean success rate: {mean_sr:.1%}")
            print(f"{'─' * 70}")

        # Save complete results
        elapsed_total = time.time() - start_time

        print(f"\n{'=' * 70}")
        print(f"ABLATION MATRIX COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total runs executed: {current_run}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {failed_runs}")
        print(f"  Success rate: {successful_runs/current_run:.1%}")
        print(f"Total elapsed time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
        print(f"Average time per run: {elapsed_total/current_run:.1f}s")
        print(f"{'=' * 70}\n")

        self._save_all_results(all_results)
        self._export_to_csv(all_results)

        return all_results

    def _save_single_result(self, result: Dict, ablation: str, seed: int):
        """Save individual run result to JSON file."""
        filepath = self.results_dir / f"{ablation}_seed{seed}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Failed to save result: {e}")

    def _save_ablation_results(self, ablation: str, results: List[Dict]):
        """Save aggregated results for one ablation across all seeds."""
        filepath = self.results_dir / f"{ablation}_aggregated.json"

        aggregated = {
            "ablation": ablation,
            "num_seeds": len(results),
            "num_successful": len([r for r in results if "error" not in r]),
            "num_failed": len([r for r in results if "error" in r]),
            "mean_metrics": self._compute_mean_metrics(results),
            "std_metrics": self._compute_std_metrics(results),
            "individual_runs": results
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
            "experiment": "ablation_study",
            "num_tasks": self.num_tasks,
            "num_seeds": self.num_seeds,
            "seeds": self.seeds,
            "ablations": list(all_results.keys()),
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
        """Export results to CSV for Table 2 in manuscript."""
        import csv

        csv_path = self.results_dir / "table2_data.csv"

        # Compute deltas relative to Full system
        full_results = all_results.get("SelfEvolve-Full", [])
        full_tta = self._compute_mean_tta([r for r in full_results if "error" not in r])

        if not full_results or full_tta == float('inf'):
            print("\n⚠️  Warning: No valid baseline results for comparison")

        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    "Configuration", "Success_Rate_Mean", "Success_Rate_Std",
                    "RFR_Mean", "RFR_Std",
                    "TTA_Mean", "TTA_Std", "Delta_TTA_vs_Full",
                    "Patches_Accepted", "Rollbacks", "Component_Removed"
                ])

                # Write data rows
                for ablation_name, results in all_results.items():
                    valid_results = [r for r in results if "error" not in r]
                    if not valid_results:
                        continue

                    # Compute statistics
                    success_rates = [r["metrics"]["success_rate"] for r in valid_results]
                    rfr_rates = [r["metrics"]["repeat_failure_rate"] for r in valid_results]
                    tta_mean = self._compute_mean_tta(valid_results)
                    tta_std = self._compute_std_tta(valid_results)

                    if HAS_NUMPY:
                        sr_mean = np.mean(success_rates)
                        sr_std = np.std(success_rates)
                        rfr_mean = np.mean(rfr_rates)
                        rfr_std = np.std(rfr_rates)
                        patches = np.mean([r["metrics"]["patches_accepted"] for r in valid_results])
                        rollbacks = np.mean([r["metrics"].get("rollback_frequency", 0) for r in valid_results])
                    else:
                        sr_mean = sum(success_rates) / len(success_rates)
                        sr_std = 0.0
                        rfr_mean = sum(rfr_rates) / len(rfr_rates)
                        rfr_std = 0.0
                        patches = sum(r["metrics"]["patches_accepted"] for r in valid_results) / len(valid_results)
                        rollbacks = sum(r["metrics"].get("rollback_frequency", 0) for r in valid_results) / len(valid_results)

                    # Compute delta TTA
                    delta_tta = tta_mean - full_tta if full_tta != float('inf') and tta_mean != float('inf') else None

                    # Extract component name
                    component_removed = ablation_name.replace("No-", "").replace("SelfEvolve-Full", "None")

                    writer.writerow([
                        ablation_name,
                        f"{sr_mean:.3f}",
                        f"{sr_std:.3f}",
                        f"{rfr_mean:.3f}",
                        f"{rfr_std:.3f}",
                        f"{tta_mean:.1f}" if tta_mean != float('inf') else "∞",
                        f"{tta_std:.1f}" if tta_std != float('inf') else "N/A",
                        f"{delta_tta:+.1f}" if delta_tta is not None else "N/A",
                        f"{patches:.1f}",
                        f"{rollbacks:.1f}",
                        component_removed
                    ])

            print(f"💾 Exported CSV: {csv_path}")

        except Exception as e:
            print(f"\n⚠️  Failed to export CSV: {e}")

    def _compute_mean_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean metrics across seeds."""
        valid = [r for r in results if "error" not in r]
        if not valid:
            return {}

        metrics = {
            "success_rate": self._safe_mean([r["metrics"]["success_rate"] for r in valid]),
            "repeat_failure_rate": self._safe_mean([r["metrics"]["repeat_failure_rate"] for r in valid]),
            "tta_mean": self._compute_mean_tta(valid),
            "patches_accepted": self._safe_mean([r["metrics"]["patches_accepted"] for r in valid]),
            "patches_proposed": self._safe_mean([r["metrics"]["patches_proposed"] for r in valid]),
        }

        return metrics

    def _compute_std_metrics(self, results: List[Dict]) -> Dict:
        """Compute standard deviation of metrics across seeds."""
        valid = [r for r in results if "error" not in r]
        if not valid or len(valid) < 2:
            return {}

        metrics = {
            "success_rate": self._safe_std([r["metrics"]["success_rate"] for r in valid]),
            "repeat_failure_rate": self._safe_std([r["metrics"]["repeat_failure_rate"] for r in valid]),
        }

        return metrics

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
        """Compute standard deviation of TTA."""
        tta_values = []

        for r in results:
            if "error" not in r:
                for v in r["metrics"]["time_to_adapt"].values():
                    if v is not None:
                        tta_values.append(v)

        return self._safe_std(tta_values) if len(tta_values) > 1 else float('inf')

    def _safe_mean(self, values: List[float]) -> float:
        """Safely compute mean with fallback."""
        if not values:
            return 0.0

        if HAS_NUMPY:
            return float(np.mean(values))
        else:
            return sum(values) / len(values)

    def _safe_std(self, values: List[float]) -> float:
        """Safely compute standard deviation with fallback."""
        if not values or len(values) < 2:
            return 0.0

        if HAS_NUMPY:
            return float(np.std(values))
        else:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5

    def _format_tta(self, tta_dict: Dict) -> str:
        """Format TTA dictionary for display."""
        if not tta_dict:
            return "N/A"

        values = []
        for k, v in tta_dict.items():
            if v is not None:
                values.append(f"{k}={v}")
            else:
                values.append(f"{k}=∞")

        return ", ".join(values) if values else "N/A"

    def print_summary(self, all_results: Dict):
        """Print comprehensive ablation study summary."""
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        # Get baseline results
        full_results = all_results.get("SelfEvolve-Full", [])
        full_valid = [r for r in full_results if "error" not in r]

        if not full_valid:
            print("\n⚠️  ERROR: No valid baseline results!")
            return

        full_tta = self._compute_mean_tta(full_valid)
        full_sr = self._safe_mean([r["metrics"]["success_rate"] for r in full_valid])

        print(f"\nBaseline (Full System):")
        print(f"  Success Rate: {full_sr:.1%}")
        print(f"  TTA: {full_tta:.1f} tasks" if full_tta != float('inf') else "  TTA: ∞")

        print(f"\nComponent Contributions (Δ vs Full System):")
        print(f"{'─' * 70}")
        print(f"{'Component':<25} {'Δ Success Rate':<20} {'Δ TTA':<15} {'Status'}")
        print(f"{'─' * 70}")

        for ablation_name, results in all_results.items():
            if ablation_name == "SelfEvolve-Full":
                continue

            valid = [r for r in results if "error" not in r]
            if not valid:
                component = ablation_name.replace("No-", "")
                print(f"{component:<25} {'N/A':<20} {'N/A':<15} ❌ Failed")
                continue

            # Compute deltas
            abl_sr = self._safe_mean([r["metrics"]["success_rate"] for r in valid])
            abl_tta = self._compute_mean_tta(valid)

            delta_sr = abl_sr - full_sr
            delta_tta = abl_tta - full_tta if abl_tta != float('inf') and full_tta != float('inf') else None

            component = ablation_name.replace("No-", "")

            sr_str = f"{delta_sr:+.1%}"
            tta_str = f"{delta_tta:+.1f} tasks" if delta_tta is not None else "N/A"
            status = "✅" if len(valid) == len(results) else "⚠️"

            print(f"{component:<25} {sr_str:<20} {tta_str:<15} {status}")

        print(f"{'─' * 70}")
        print(f"\nInterpretation:")
        print(f"  Negative Δ Success Rate = Component removal hurts performance")
        print(f"  Positive Δ TTA = Component removal slows adaptation")
        print(f"  ∞ TTA = No adaptation occurred (system cannot learn)")

        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {self.results_dir}")
        print(f"✓ CSV data for Table 2: {self.results_dir}/table2_data.csv")
        print("=" * 70)


def main():
    """Main entry point for ablation study."""
    try:
        experiment = AblationStudy()
        all_results = experiment.run_all_ablations()
        experiment.print_summary(all_results)

        print("\n✓ Ablation study complete!")
        print(f"   Check {experiment.results_dir}/table2_data.csv for table data")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        print("   Partial results may be saved in experiments/results/ablations/")

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())