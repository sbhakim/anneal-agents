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
- CRITICAL FIX: Deep merge for proper nested config overrides
- ADDED: Debug mode to verify config differences
- CRITICAL FIX (NEW): Stop using 'fdka.threshold=999.0' as a kill-switch; use 'fdka.enabled: false'
- CRITICAL FIX (NEW): Validation layer auto-normalizes legacy overrides (e.g., converts threshold-kill-switch to boolean)
- CRITICAL FIX (NEW): Ensure ablation runs have a safe offline provider/model unless explicitly set
- ADDED (NEW): Provider/model echo in debug to catch misconfig quickly
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import copy
import argparse  # ✅ CRITICAL FIX: Added missing import

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


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge override dictionary into base dictionary.
    ✅ CRITICAL FIX: This ensures nested config values are properly overridden.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Direct override
            result[key] = copy.deepcopy(value)

    return result


def _normalize_fdka_overrides(ov: Dict) -> Dict:
    """
    Normalize any FDKA-related overrides to avoid 'threshold-as-off' patterns.
    - If 'fdka.threshold' is set to an extreme value (e.g., >= 900), convert to {'fdka': {'enabled': False}}
    - Preserve explicit threshold if it's a reasonable numeric (e.g., < 10).
    """
    if not isinstance(ov, dict):
        return ov

    # Work on a deep copy to be safe
    out = copy.deepcopy(ov)

    fdka_cfg = out.get("fdka", {})
    if isinstance(fdka_cfg, dict):
        # If legacy kill-switch is present, normalize it
        th = fdka_cfg.get("threshold", None)
        en = fdka_cfg.get("enabled", None)
        if th is not None and isinstance(th, (int, float)) and th >= 900:
            # Convert to explicit disable
            fdka_cfg.pop("threshold", None)
            fdka_cfg["enabled"] = False
            out["fdka"] = fdka_cfg
        elif en is None:
            # If 'enabled' not provided, leave as-is; runtime will default to True from base config
            out["fdka"] = fdka_cfg

    return out


def _validate_and_normalize_overrides(config_overrides: Dict, ablation_name: str, debug_mode: bool) -> Dict:
    """
    Validate and normalize overrides before applying:
    - Ensure No-FDKA uses boolean 'enabled: false' instead of threshold hacks.
    - Log warnings when legacy patterns are detected and auto-corrected.
    """
    original = copy.deepcopy(config_overrides)
    normalized = copy.deepcopy(config_overrides)

    # Normalize FDKA overrides across all ablations
    normalized = _normalize_fdka_overrides(normalized)

    # Special case: enforce boolean disable for No-FDKA
    if ablation_name == "No-FDKA":
        fdka = normalized.setdefault("fdka", {})
        # Force explicit boolean gate; do not rely on threshold hacks
        fdka["enabled"] = False
        # Remove any residual threshold-kill-switch remnants
        if "threshold" in fdka and isinstance(fdka["threshold"], (int, float)) and fdka["threshold"] >= 900:
            fdka.pop("threshold", None)

    # Debug/Warning output if we changed anything
    if debug_mode and original != normalized:
        print("\n⚙️  NORMALIZED OVERRIDES (auto-corrections applied):")
        print("   Original:", json.dumps(original, indent=2))
        print("   Normalized:", json.dumps(normalized, indent=2))

    # Lightweight validation: ensure we didn't end up with contradictory settings
    fdka = normalized.get("fdka", {})
    if "enabled" in fdka and isinstance(fdka.get("enabled"), bool) and fdka["enabled"] is False:
        # If explicitly disabled, do not allow a tiny threshold suggesting it's enabled
        if "threshold" in fdka and isinstance(fdka["threshold"], (int, float)) and fdka["threshold"] < 10:
            if debug_mode:
                print("   🛑 Adjusting contradictory FDKA settings: 'enabled: false' with small 'threshold'. Removing threshold.")
            fdka.pop("threshold", None)
            normalized["fdka"] = fdka

    return normalized


def _ensure_offline_provider(experiment_config: Dict, debug_mode: bool) -> None:
    """
    Ensure a safe offline provider/model exists for FDKA propose_edit when enabled.
    - Uses setdefault so explicit user selections are respected.
    - Prevents ablation baseline from failing due to missing API keys.
    """
    fdka_cfg = experiment_config.setdefault("fdka", {})
    if fdka_cfg.get("enabled", True):
        pe = fdka_cfg.setdefault("propose_edit", {})
        # Only set defaults if not provided by config/overrides
        provider_before = pe.get("llm_provider")
        model_before = pe.get("model")
        pe.setdefault("llm_provider", "transformers")
        pe.setdefault("model", "mistralai/Mistral-7B-Instruct-v0.3")
        if debug_mode:
            print("🔧 Provider defaults applied (setdefault):")
            print(f"   llm_provider: {provider_before} → {pe.get('llm_provider')}")
            print(f"   model       : {model_before} → {pe.get('model')}")


class AblationStudy:
    """
    Ablation experiments to measure component contributions.
    Systematically disables one component at a time.
    """

    def __init__(self, base_config_path: str = "config.yaml", debug_mode: bool = False):
        """
        Initialize ablation study.
        """
        config_path = Path(__file__).parent.parent / base_config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.base_config = load_config(str(config_path))
        self.debug_mode = debug_mode

        self.results_dir = Path("experiments/results/ablations")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.num_seeds = 3
        self.seeds = [42, 123, 456]

        # This line correctly focuses the study on the 'hard' difficulty
        self.difficulty_levels = ['hard']
        # self.difficulty_levels = ['easy', 'normal', 'hard']
        self.task_counts = {'easy': 30, 'normal': 50, 'hard': 20}

        print("=" * 70)
        print("ABLATION STUDY EXPERIMENT (COMPONENT STRESS TEST)")
        print("=" * 70)
        print(f"Configuration: {len(self.difficulty_levels)} difficulties × {self.num_seeds} seeds")
        print(f"Results directory: {self.results_dir}")
        print(f"Base config loaded from: {config_path}")
        print(f"Debug mode: {'ENABLED' if debug_mode else 'DISABLED'}")
        print()

    def _run_output_dir(self, ablation_name: str, seed: int, difficulty: str) -> Path:
        """
        Per-run output dir to avoid overwriting data/results across ablations.
        Keeps paths deterministic for easy aggregation.
        """
        safe_name = ablation_name.replace(" ", "_").replace("/", "_")
        return self.results_dir / "runs" / safe_name / difficulty / f"seed{seed}"

    def get_ablation_configs(self) -> Dict[str, Dict]:
        """
        Define ablation configurations.
        Each disables one key component to measure its contribution.
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
                        "trust": {"alpha": 1, "beta": 1},
                        "gates": {
                            # keep legacy knobs for backwards-compat, but explicitly disable hard gates
                            "tau_impact": 999.0,
                            "tau_conf": 0.0,
                            "use_z3": False,
                            "z3_min_coverage": 1.0,
                            "z3_unknown_policy": "allow",
                        },
                        "canary": {"num_tests": 0},
                        "rollback": {"enable": False},
                    }
                }
            },
            "No-Verify": {
                "name": "Without Verify-Before-Act",
                "description": "FDKA + governance, but no precondition pre-checking",
                "config_overrides": {
                    "executor": {"enable_verification": False}
                }
            },
            "No-Arbitration": {
                "name": "Without Metacognitive Arbitration",
                "description": "Always use S1 pathway, no adaptive routing",
                "config_overrides": {
                    "metacognition": {
                        "tau_u": 999.0,
                        "tau_p": 999.0,
                        "enable_reflection": False
                    }
                }
            },
            "No-FDKA": {
                "name": "Without FDKA",
                "description": "Verify + Arbitration but no self-evolution (static operators)",
                "config_overrides": {
                    # ✅ CRITICAL FIX: Do NOT use threshold as a kill-switch.
                    # Use an explicit boolean gate recognized by the runtime.
                    "fdka": {"enabled": False}
                }
            }
        }
        return configs

    def run_single_ablation(self, ablation_name: str, config: Dict, seed: int, difficulty: str) -> Dict:
        """
        Run one ablation configuration with a specific seed and difficulty.
        """
        print(f"\n{'─' * 70}")
        print(f"Running: {ablation_name} (seed={seed}, difficulty={difficulty})")
        print(f"{'─' * 70}")

        experiment_config = copy.deepcopy(self.base_config)
        raw_overrides = config.get("config_overrides", {})

        # Validate & normalize overrides (prevents threshold-as-off)
        config_overrides = _validate_and_normalize_overrides(raw_overrides, ablation_name, self.debug_mode)

        if self.debug_mode and config_overrides:
            print(f"\n🔍 BEFORE applying overrides:")
            print(f"   FDKA enabled: {experiment_config.get('fdka', {}).get('enabled', 'N/A')}")
            print(f"   FDKA threshold: {experiment_config.get('fdka', {}).get('threshold', 'N/A')}")
            print(f"   Governance enabled: {experiment_config.get('governance', {}).get('provenance', {}).get('enable', 'N/A')}")
            print(f"   Verify enabled: {experiment_config.get('executor', {}).get('enable_verification', 'N/A')}")
            print(f"   Metacog tau_u: {experiment_config.get('metacognition', {}).get('tau_u', 'N/A')}")
            pe0 = experiment_config.get('fdka', {}).get('propose_edit', {})
            print(f"   Provider/model: {pe0.get('llm_provider')} / {pe0.get('model')}")

        if config_overrides:
            experiment_config = deep_merge(experiment_config, config_overrides)

        # ✅ Ensure offline-safe provider/model defaults so baselines don't vanish due to API keys
        _ensure_offline_provider(experiment_config, self.debug_mode)

        if self.debug_mode:
            print(f"\n🔍 AFTER applying overrides & defaults:")
            print(f"   FDKA enabled: {experiment_config.get('fdka', {}).get('enabled', 'N/A')}")
            print(f"   FDKA threshold: {experiment_config.get('fdka', {}).get('threshold', 'N/A')}")
            print(f"   Governance enabled: {experiment_config.get('governance', {}).get('provenance', {}).get('enable', 'N/A')}")
            print(f"   Verify enabled: {experiment_config.get('executor', {}).get('enable_verification', 'N/A')}")
            print(f"   Metacog tau_u: {experiment_config.get('metacognition', {}).get('tau_u', 'N/A')}")
            pe1 = experiment_config.get('fdka', {}).get('propose_edit', {})
            print(f"   Provider/model: {pe1.get('llm_provider')} / {pe1.get('model')}")
            print(f"   Overrides applied: {list(config_overrides.keys()) if config_overrides else []}")

        experiment_config['scenario']['difficulty'] = difficulty
        experiment_config['scenario']['num_tasks'] = self.task_counts[difficulty]
        experiment_config['scenario']['failure_injector_seed'] = seed
        experiment_config['logging']['level'] = 'WARNING'

        # Per-run isolation: avoid overwriting data/results and mixing provenance across ablations
        run_dir = self._run_output_dir(ablation_name, seed, difficulty)
        run_dir.mkdir(parents=True, exist_ok=True)
        experiment_config.setdefault("output", {})
        experiment_config["output"]["results_dir"] = str(run_dir)
        # Keep plots under the same run dir (if consumed by code)
        experiment_config["output"]["plots_dir"] = str(run_dir / "plots")

        # Isolate log file per run for auditability
        experiment_config.setdefault("logging", {})
        experiment_config["logging"]["log_file"] = str(run_dir / "selfevolve.log")

        # Isolate provenance log per run when enabled
        gov = experiment_config.setdefault("governance", {})
        prov = gov.setdefault("provenance", {})
        if prov.get("enable", True):
            prov["log_path"] = str(run_dir / "provenance.jsonl")

        start_time = time.time()
        try:
            system = SelfEvolveSystem(experiment_config)
            metrics = system.run_evaluation()
            elapsed = time.time() - start_time
            summary = metrics.get_summary() if hasattr(metrics, 'get_summary') else metrics
            result = {
                "ablation": ablation_name, "seed": seed, "difficulty": difficulty,
                "elapsed_time": elapsed, "metrics": summary, "config": config,
                "timestamp": time.time(), "run_dir": str(run_dir)
            }
            print(f"\n✓ Completed in {elapsed:.1f}s")
            print(f"  Success Rate: {summary['success_rate']:.1%}")
            print(f"  RFR: {summary['repeat_failure_rate']:.1%}")
            print(f"  Patches Accepted: {summary.get('patches_accepted', 0)}")
            print(f"  TTA: {self._format_tta(summary['time_to_adapt'])}")
            return result
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "ablation": ablation_name, "seed": seed, "difficulty": difficulty,
                "error": str(e), "traceback": traceback.format_exc(),
                "timestamp": time.time(), "run_dir": str(run_dir)
            }

    def run_all_ablations(self) -> Dict[str, List[Dict]]:
        """
        Run complete ablation matrix across all difficulties and seeds.
        """
        configs = self.get_ablation_configs()
        all_results = {}
        total_runs = len(configs) * len(self.seeds) * len(self.difficulty_levels)
        current_run = 0

        print(f"\n{'=' * 70}\nSTARTING ABLATION MATRIX\n{'=' * 70}")
        print(f"Total configurations: {len(configs)}")
        print(f"Seeds per config: {len(self.seeds)}")
        print(f"Difficulty levels: {self.difficulty_levels}")
        print(f"Total runs: {total_runs}")
        print(f"\nAblations to run:")
        for i, (name, cfg) in enumerate(configs.items(), 1):
            print(f"  {i}. {name}: {cfg['description']}")
        print(f"\nSeeds: {self.seeds}\n{'=' * 70}\n")

        start_time = time.time()
        for ablation_name, config in configs.items():
            ablation_results = []
            for difficulty in self.difficulty_levels:
                for seed in self.seeds:
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
        print(f"\n{'=' * 70}\nABLATION MATRIX COMPLETE\n{'=' * 70}")
        successful_runs = sum(1 for res in all_results.values() for r in res if "error" not in r)
        print(f"Total runs executed: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"Total elapsed time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)\n{'=' * 70}\n")

        self._save_all_results(all_results)
        self._export_to_csv(all_results)
        return all_results

    def print_summary(self, all_results: Dict[str, List[Dict]]):
        """
        Print formatted summary of ablation study results.
        """
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        full_system_results = all_results.get('SelfEvolve-Full', [])
        valid_baseline = [r for r in full_system_results if "error" not in r]

        if not valid_baseline:
            print("⚠️  No valid baseline runs found to compare against.")
            return

        baseline_sr = self._safe_mean([r['metrics']['success_rate'] for r in valid_baseline])
        baseline_rfr = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_baseline])
        baseline_tta = self._compute_mean_tta(valid_baseline)
        baseline_patches = self._safe_mean([r['metrics'].get('patches_accepted', 0) for r in valid_baseline])

        print(f"\nBaseline (Full System):")
        print(f"  Success Rate: {baseline_sr:.1%}")
        print(f"  RFR: {baseline_rfr:.1%}")
        print(f"  TTA: {baseline_tta:.1f} tasks")
        print(f"  Patches Accepted: {baseline_patches:.1f}")

        print(f"\nComponent Contributions (Δ vs Full System):")
        print("-" * 80)
        print(f"{'Component':<20} {'Δ SR':>10} {'Δ RFR':>10} {'Δ TTA':>10} {'Δ Patches':>12} {'Status':>8}")
        print("-" * 80)

        for ablation_name, results in all_results.items():
            if ablation_name == 'SelfEvolve-Full': continue
            component = ablation_name.replace('No-', '')
            valid_runs = [r for r in results if "error" not in r]
            if not valid_runs:
                print(f"{component:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'❌':>8}")
                continue

            sr = self._safe_mean([r['metrics']['success_rate'] for r in valid_runs])
            rfr = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_runs])
            tta = self._compute_mean_tta(valid_runs)
            patches = self._safe_mean([r['metrics'].get('patches_accepted', 0) for r in valid_runs])

            delta_sr = (sr - baseline_sr) * 100
            delta_rfr = (rfr - baseline_rfr) * 100
            delta_tta = tta - baseline_tta if tta != float('inf') and baseline_tta != float('inf') else float('inf')
            delta_patches = patches - baseline_patches

            status = "✅"
            if abs(delta_sr) < 1 and abs(delta_rfr) < 1 and abs(delta_tta) < 1 and abs(delta_patches) < 0.5:
                status = "⚠️"
            elif delta_sr < -2 or delta_rfr > 2 or (isinstance(delta_tta, float) and delta_tta > 2) or delta_patches < -0.5:
                status = "❌"

            tta_str = f"{delta_tta:+.1f}" if delta_tta != float('inf') else "∞"
            print(f"{component:<20} {delta_sr:>+9.1f}% {delta_rfr:>+9.1f}% {tta_str:>10} {delta_patches:>+11.1f} {status:>8}")

        print("-" * 80)
        print("\n💡 Interpretation:")
        print("   • ❌: Component removal significantly hurts performance (this is expected and good!)")
        print("   • ✅: Component removal has a minor or expected impact.")
        print("   • ⚠️: All deltas are near zero, indicating config overrides may not be working or scenario is too easy.")
        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {self.results_dir}")
        print(f"✓ CSV data for Table 2: {self.results_dir / 'table2_data.csv'}")
        print("=" * 70)

    def _save_single_result(self, result: Dict, ablation: str, seed: int, difficulty: str):
        filepath = self.results_dir / f"{ablation}_{difficulty}_seed{seed}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Failed to save result: {e}")

    def _save_ablation_results(self, ablation: str, results: List[Dict]):
        filepath = self.results_dir / f"{ablation}_aggregated.json"
        aggregated = {"ablation": ablation, "num_runs": len(results), "runs": results, "timestamp": time.time()}
        try:
            with open(filepath, 'w') as f:
                json.dump(aggregated, f, indent=2)
            print(f"  💾 Saved aggregated results: {filepath.name}")
        except Exception as e:
            print(f"  ⚠️  Failed to save aggregated results: {e}")

    def _save_all_results(self, all_results: Dict):
        filepath = self.results_dir / "complete_ablations.json"
        summary = {
            "experiment": "ablation_study_stress_test", "seeds": self.seeds, "difficulties": self.difficulty_levels,
            "task_counts": self.task_counts, "num_configurations": len(all_results),
            "total_runs": sum(len(results) for results in all_results.values()),
            "results": all_results, "timestamp": time.time()
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Saved complete ablations: {filepath}")
        except Exception as e:
            print(f"\n⚠️  Failed to save complete results: {e}")

    def _export_to_csv(self, all_results: Dict):
        import csv
        csv_path = self.results_dir / "table2_data.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Configuration", "Difficulty", "Success_Rate_Mean", "Success_Rate_Std",
                    "RFR_Mean", "RFR_Std", "RFR_Observed_Mean", "RFR_Observed_Std",
                    "RFR_Terminal_Mean", "RFR_Terminal_Std", "TTA_Mean", "TTA_Std",
                    "Patches_Accepted", "Rollbacks", "Component_Removed"
                ])
                for ablation_name, results in all_results.items():
                    for difficulty in self.difficulty_levels:
                        valid_runs = [r for r in results if "error" not in r and r.get('difficulty') == difficulty]
                        if not valid_runs: continue
                        sr_mean = self._safe_mean([r['metrics']['success_rate'] for r in valid_runs])
                        sr_std = self._safe_std([r['metrics']['success_rate'] for r in valid_runs])
                        rfr_mean = self._safe_mean([r['metrics']['repeat_failure_rate'] for r in valid_runs])
                        rfr_std = self._safe_std([r['metrics']['repeat_failure_rate'] for r in valid_runs])
                        rfr_obs_mean = self._safe_mean([
                            r['metrics'].get('observed_rfr', r['metrics'].get('repeat_failure_rate', 0.0))
                            for r in valid_runs
                        ])
                        rfr_obs_std = self._safe_std([
                            r['metrics'].get('observed_rfr', r['metrics'].get('repeat_failure_rate', 0.0))
                            for r in valid_runs
                        ])
                        rfr_term_mean = self._safe_mean([
                            r['metrics'].get('terminal_rfr', 0.0)
                            for r in valid_runs
                        ])
                        rfr_term_std = self._safe_std([
                            r['metrics'].get('terminal_rfr', 0.0)
                            for r in valid_runs
                        ])
                        tta_mean = self._compute_mean_tta(valid_runs)
                        tta_std = self._compute_std_tta(valid_runs)
                        patches = self._safe_mean([r['metrics'].get('patches_accepted', 0) for r in valid_runs])
                        rollbacks = self._safe_mean([r['metrics'].get('rollback_frequency', 0) for r in valid_runs])
                        component = ablation_name.replace("No-", "").replace("SelfEvolve-Full", "None")
                        writer.writerow([
                            ablation_name, difficulty, f"{sr_mean:.3f}", f"{sr_std:.3f}", f"{rfr_mean:.3f}",
                            f"{rfr_std:.3f}", f"{rfr_obs_mean:.3f}", f"{rfr_obs_std:.3f}",
                            f"{rfr_term_mean:.3f}", f"{rfr_term_std:.3f}", f"{tta_mean:.1f}" if tta_mean != float('inf') else "∞",
                            f"{tta_std:.1f}" if tta_std != float('inf') else "∞", f"{patches:.1f}",
                            f"{rollbacks:.1f}", component
                        ])
            print(f"💾 Exported CSV: {csv_path}")
        except Exception as e:
            print(f"\n⚠️  Failed to export CSV: {e}")

    def _compute_mean_tta(self, results: List[Dict]) -> float:
        tta_values = [v for r in results if "error" not in r for v in r["metrics"]["time_to_adapt"].values() if v is not None]
        return self._safe_mean(tta_values) if tta_values else float('inf')

    def _compute_std_tta(self, results: List[Dict]) -> float:
        tta_values = [v for r in results if "error" not in r for v in r["metrics"]["time_to_adapt"].values() if v is not None]
        return self._safe_std(tta_values) if len(tta_values) >= 2 else float('inf')

    def _safe_mean(self, values: List[float]) -> float:
        if not values: return 0.0
        return sum(values) / len(values) if not HAS_NUMPY else float(np.mean(values))

    def _safe_std(self, values: List[float]) -> float:
        if not values or len(values) < 2: return 0.0
        if HAS_NUMPY: return float(np.std(values))
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    def _format_tta(self, tta_dict: Dict) -> str:
        if not tta_dict: return "N/A"
        values = [f"{k}={v}" if v is not None else f"{k}=∞" for k, v in tta_dict.items()]
        return ", ".join(values) if values else "N/A"


def main():
    """Main entry point for ablation study."""
    parser = argparse.ArgumentParser(description='Run ablation study experiments')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--quick-test', action='store_true', help='Run quick 2-config test')
    args = parser.parse_args()

    try:
        experiment = AblationStudy(debug_mode=args.debug)
        if args.quick_test:
            # Minimal quick test: run Full and No-FDKA on one seed/difficulty for smoke-check
            cfgs = experiment.get_ablation_configs()
            subset = {k: cfgs[k] for k in ["SelfEvolve-Full", "No-FDKA"] if k in cfgs}
            all_results = {}
            for name, cfg in subset.items():
                res = []
                # fixed seed/difficulty
                res.append(experiment.run_single_ablation(name, cfg, seed=42, difficulty='hard'))
                all_results[name] = res
            experiment._save_all_results(all_results)
            experiment._export_to_csv(all_results)
            experiment.print_summary(all_results)
            return 0

        all_results = experiment.run_all_ablations()
        experiment.print_summary(all_results)
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
