#!/usr/bin/env python3
# experiments/run_full_evaluation.py
"""
Master Evaluation Script for ACM Manuscript
============================================

Runs all experiments required for manuscript data generation:
- Baseline Comparison → Table 1 (Main Results)
- Ablation Study → Table 2 (Component Contributions)
- Demo Run → Tables 3-5 (Per-Failure Analysis, Governance, Efficiency)

Usage:
    python experiments/run_full_evaluation.py [--quick] [--skip-baseline] [--skip-ablations]

Options:
    --quick           Run with reduced settings (e.g., 10 tasks, 2 seeds) for testing
    --skip-baseline   Skip baseline comparison experiment
    --skip-ablations  Skip ablation study experiment
    --demo-only       Run only a single demo run for qualitative analysis
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.core.system import SelfEvolveSystem
from experiments.run_baseline_comparison import BaselineComparison
from experiments.run_ablations import AblationStudy
from experiments.run_governance_stress import run_stress as run_governance_stress


class FullEvaluationRunner:
    """
    Orchestrates all experiments for manuscript data generation.
    """

    def __init__(self, quick_mode: bool = False):
        """
        Initialize evaluation runner.
        """
        self.quick_mode = quick_mode
        self.results_root = Path("experiments/results")
        self.data_root = Path("data/results")
        self.start_time = time.time()

        # Stable orchestration run id (keeps logs + manifests joinable)
        self.run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.run_dir = self.results_root / "full_runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Track experiment results
        self.experiment_results = {
            "baseline_comparison": {"status": "not_run", "duration": 0, "error": None},
            "ablation_study": {"status": "not_run", "duration": 0, "error": None},
            "demo_run": {"status": "not_run", "duration": 0, "error": None},
            "governance_stress": {"status": "not_run", "duration": 0, "error": None},
        }

        # ✅ Initialize experiment objects once
        self.baseline_runner = BaselineComparison()
        self.ablation_runner = AblationStudy()

        # ------------------------------------------------------------------
        # Per-orchestration isolation
        # ------------------------------------------------------------------
        # BaselineComparison and AblationStudy default to shared global folders under
        # experiments/results/*. That makes repeated runs overwrite/mix artifacts and
        # (for baselines) may also write logs/provenance to shared paths.
        # We re-root both experiments under this orchestration run directory so
        # every run is self-contained and auditable.

        self.baseline_dir = self.run_dir / "baseline_comparison"
        self.ablation_dir = self.run_dir / "ablations"
        self.gov_stress_dir = self.run_dir / "governance_stress"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.ablation_dir.mkdir(parents=True, exist_ok=True)
        self.gov_stress_dir.mkdir(parents=True, exist_ok=True)

        # Re-root experiment-level result aggregation outputs (JSON/CSV).
        self.baseline_runner.results_dir = self.baseline_dir
        self.baseline_runner.results_dir.mkdir(parents=True, exist_ok=True)
        self.ablation_runner.results_dir = self.ablation_dir
        self.ablation_runner.results_dir.mkdir(parents=True, exist_ok=True)

        if self.quick_mode:
            print("⚡ QUICK MODE: Modifying experiment settings for a fast run.")
            # Baselines
            self.baseline_runner.num_tasks = 10
            self.baseline_runner.num_seeds = 2
            self.baseline_runner.seeds = [42, 123]
            # Ablations
            self.ablation_runner.num_seeds = 2
            self.ablation_runner.seeds = [42, 123]
            self.ablation_runner.task_counts = {'easy': 10, 'normal': 10, 'hard': 10}
            # Keep ablation difficulty list as defined by the runner; task_counts above will apply.

        self.results_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("FULL EVALUATION FOR ACM MANUSCRIPT")
        print("=" * 70)
        print(f"Mode: {'QUICK TEST' if quick_mode else 'FULL EVALUATION'}")
        print(f"Run ID: {self.run_id}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_root}")
        print(f"Orchestration Run Dir: {self.run_dir}")
        print(f"Baseline Artifacts Dir: {self.baseline_dir}")
        print(f"Ablation Artifacts Dir: {self.ablation_dir}")
        print(f"Governance Stress Dir: {self.gov_stress_dir}")
        print("=" * 70)
        print()

    def _read_json_if_exists(self, path: Path):
        """Best-effort JSON loader for manifests; returns None when missing/unreadable."""
        try:
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            return None
        return None

    def _baseline_run_output_dir(self, system_name: str, seed: int) -> Path:
        """Per-run output dir for baseline runs (prevents overwrite across seeds/systems)."""
        safe_name = system_name.replace(" ", "_").replace("/", "_")
        return self.baseline_dir / "runs" / safe_name / f"seed{seed}"

    def _prepare_baseline_base_config_for_run(self, run_dir: Path) -> None:
        """
        Mutate BaselineComparison.base_config so each run writes logs/metrics/provenance
        into the given run_dir. BaselineComparison.run_single_experiment() deep-copies
        base_config, so this is safe per run.
        """
        cfg = self.baseline_runner.base_config

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)

        cfg.setdefault("output", {})
        cfg["output"]["results_dir"] = str(run_dir)
        cfg["output"]["plots_dir"] = str(run_dir / "plots")

        cfg.setdefault("logging", {})
        cfg["logging"]["log_file"] = str(run_dir / "selfevolve.log")

        gov = cfg.setdefault("governance", {})
        prov = gov.setdefault("provenance", {})
        if prov.get("enable", True):
            prov["log_path"] = str(run_dir / "provenance.jsonl")

    def _run_baseline_comparison_isolated(self) -> dict:
        """
        Run the baseline matrix like BaselineComparison.run_all_experiments(), but with
        per-run config isolation so metrics/logs/provenance never overwrite each other.
        """
        configs = self.baseline_runner.get_system_configs()
        all_results = {}

        total_runs = len(configs) * len(self.baseline_runner.seeds)
        current_run = 0

        print(f"\nStarting {total_runs} experiment runs (isolated)...")
        print(f"Systems: {list(configs.keys())}")
        print(f"Seeds: {self.baseline_runner.seeds}\n")

        for system_name, config in configs.items():
            system_results = []
            for seed in self.baseline_runner.seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {system_name} × seed={seed}")

                run_dir = self._baseline_run_output_dir(system_name, seed)
                self._prepare_baseline_base_config_for_run(run_dir)

                result = self.baseline_runner.run_single_experiment(system_name, config, seed)
                system_results.append(result)

                # Save individual result immediately (checkpoint) under the isolated baseline_dir.
                self.baseline_runner._save_single_result(result, system_name, seed)

                # Per-run manifest for auditability/debugging
                manifest = {
                    "run_id": self.run_id,
                    "experiment": "baseline_comparison",
                    "system": system_name,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "timestamp": time.time(),
                    "config_overrides": config.get("config_overrides", {}),
                }
                try:
                    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                except Exception:
                    # Non-fatal: the experiment result JSON is the ground truth
                    pass

            all_results[system_name] = system_results
            self.baseline_runner._save_system_results(system_name, system_results)

        self.baseline_runner._save_all_results(all_results)
        return all_results

    def _write_experiment_summary(self) -> None:
        """
        Persist orchestration-level summary for reviewer auditability.
        - Includes status/duration/errors + pointers to demo run manifests if present.
        """
        total_duration = time.time() - self.start_time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        demo_dir = self.run_dir / "demo"
        demo_manifest = self._read_json_if_exists(demo_dir / "run_manifest.json")
        demo_config = self._read_json_if_exists(demo_dir / "config_resolved.json")

        baseline_complete = self._read_json_if_exists(self.baseline_dir / "complete_results.json")
        ablation_complete = self._read_json_if_exists(self.ablation_dir / "complete_ablations.json")
        gov_stress_summary = self._read_json_if_exists(self.gov_stress_dir / "governance_stress_summary.json")

        baseline_meta = None
        if isinstance(baseline_complete, dict):
            try:
                results = baseline_complete.get("results", {}) or {}
                baseline_meta = {
                    "num_tasks": baseline_complete.get("num_tasks"),
                    "seeds": baseline_complete.get("seeds"),
                    "systems": baseline_complete.get("systems") or list(results.keys()),
                    "total_runs": sum(len(v or []) for v in results.values()),
                }
            except Exception:
                baseline_meta = None

        ablation_meta = None
        if isinstance(ablation_complete, dict):
            try:
                results = ablation_complete.get("results", {}) or {}
                ablation_meta = {
                    "seeds": ablation_complete.get("seeds"),
                    "difficulties": ablation_complete.get("difficulties"),
                    "task_counts": ablation_complete.get("task_counts"),
                    "configurations": list(results.keys()),
                    "total_runs": sum(len(v or []) for v in results.values()),
                }
            except Exception:
                ablation_meta = None

        summary = {
            "run_id": self.run_id,
            "timestamp": now,
            "quick_mode": bool(self.quick_mode),
            "total_duration_seconds": total_duration,
            "experiment_results": self.experiment_results,
            "paths": {
                "results_root": str(self.results_root),
                "data_root": str(self.data_root),
                "run_dir": str(self.run_dir),
            },
            "artifacts": {
                "baseline_comparison": {
                    "dir": str(self.baseline_dir),
                    "complete_results": str(self.baseline_dir / "complete_results.json"),
                    "table_csv": str(self.baseline_dir / "table1_data.csv"),
                    "snapshot": baseline_meta,
                },
                "ablation_study": {
                    "dir": str(self.ablation_dir),
                    "complete_results": str(self.ablation_dir / "complete_ablations.json"),
                    "table_csv": str(self.ablation_dir / "table2_data.csv"),
                    "snapshot": ablation_meta,
                },
                "governance_stress": {
                    "dir": str(self.gov_stress_dir),
                    "summary_json": str(self.gov_stress_dir / "governance_stress_summary.json"),
                    "results_csv": str(self.gov_stress_dir / "governance_stress_results.csv"),
                    "table4_governance_csv": str(self.gov_stress_dir / "table4_governance.csv"),
                    "table4_governance_detail_csv": str(self.gov_stress_dir / "table4_governance_detail.csv"),
                    "snapshot": gov_stress_summary.get("governance_stats") if isinstance(gov_stress_summary, dict) else None,
                },
                "demo_run": {
                    "dir": str(self.run_dir / "demo"),
                },
            },
            # Inline the demo run manifest snapshots when available (small + helpful for debugging).
            "demo_snapshot": {
                "run_manifest": demo_manifest,
                "config_resolved": demo_config,
            },
        }

        out_path = self.run_dir / "experiment_summary.json"
        try:
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"\n💾 Wrote orchestration summary: {out_path}")
        except Exception as e:
            print(f"\n⚠️ Failed to write orchestration summary ({out_path}): {e}")

    def run_baseline_comparison(self) -> bool:
        """
        Run baseline comparison experiment (Table 1).

        NOTE:
        BaselineComparison.run_all_experiments() does not assign a unique output/log/provenance
        directory per seed/system, so multiple runs can overwrite each other and leak artifacts.
        We run an isolated loop here that re-roots base_config output paths per run.
        """
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT 1: BASELINE COMPARISON (Table 1)")
        print("=" * 70)
        experiment_start = time.time()
        self.experiment_results["baseline_comparison"]["status"] = "running"

        try:
            print("🚀 Starting baseline comparison...")
            print(f"   Writing aggregated artifacts to: {self.baseline_dir}")
            all_results = self._run_baseline_comparison_isolated()
            self.baseline_runner.print_summary(all_results)
            self.experiment_results["baseline_comparison"]["status"] = "success"
            return True

        except Exception as e:
            print(f"\n❌ Baseline comparison failed: {e}")
            self.experiment_results["baseline_comparison"]["status"] = "failed"
            self.experiment_results["baseline_comparison"]["error"] = str(e)
            return False
        finally:
            self.experiment_results["baseline_comparison"]["duration"] = time.time() - experiment_start

    def run_ablation_study(self) -> bool:
        """
        Run ablation study (Table 2).
        """
        print("\n" + "=" * 70)
        print("🔬 EXPERIMENT 2: ABLATION STUDY (Table 2)")
        print("=" * 70)
        experiment_start = time.time()
        self.experiment_results["ablation_study"]["status"] = "running"

        try:
            print("🚀 Starting ablation study...")
            print(f"   Writing aggregated artifacts to: {self.ablation_dir}")
            all_results = self.ablation_runner.run_all_ablations()
            self.ablation_runner.print_summary(all_results)
            self.experiment_results["ablation_study"]["status"] = "success"
            return True

        except Exception as e:
            print(f"\n❌ Ablation study failed: {e}")
            self.experiment_results["ablation_study"]["status"] = "failed"
            self.experiment_results["ablation_study"]["error"] = str(e)
            return False
        finally:
            self.experiment_results["ablation_study"]["duration"] = time.time() - experiment_start

    def run_demo(self) -> bool:
        """
        Run demo mode for detailed per-failure analysis (Tables 3-5).

        NOTE: We isolate demo output under experiments/results/full_runs/<run_id>/demo
        so it never overwrites other runs (or prior demo results).
        """
        print("\n" + "=" * 70)
        print("🎯 EXPERIMENT 3: DEMO RUN (Tables 3-5)")
        print("=" * 70)
        experiment_start = time.time()
        self.experiment_results["demo_run"]["status"] = "running"

        try:
            project_root = Path(__file__).parent.parent
            config = load_config(str(project_root / "config.yaml"))

            if self.quick_mode:
                config['scenario']['num_tasks'] = 10

            # Per-run isolation: demo writes its own metrics/manifests/log/provenance.
            demo_dir = self.run_dir / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)

            config.setdefault("output", {})
            config["output"]["results_dir"] = str(demo_dir)
            config["output"]["plots_dir"] = str(demo_dir / "plots")

            config.setdefault("logging", {})
            config["logging"]["log_file"] = str(demo_dir / "selfevolve.log")

            gov = config.setdefault("governance", {})
            prov = gov.setdefault("provenance", {})
            if prov.get("enable", True):
                prov["log_path"] = str(demo_dir / "provenance.jsonl")

            system = SelfEvolveSystem(config)
            system.run_evaluation()

            self.experiment_results["demo_run"]["status"] = "success"
            return True

        except Exception as e:
            print(f"\n❌ Demo run failed: {e}")
            self.experiment_results["demo_run"]["status"] = "failed"
            self.experiment_results["demo_run"]["error"] = str(e)
            return False
        finally:
            self.experiment_results["demo_run"]["duration"] = time.time() - experiment_start

    def generate_summary(self) -> None:
        """
        Generate comprehensive summary of all experiments.
        """
        print("\n" + "=" * 70)
        print("📋 EVALUATION SUMMARY")
        print("=" * 70)
        total_duration = time.time() - self.start_time
        print(f"Total Elapsed Time: {total_duration:.1f}s ({total_duration / 60:.1f} min)\n")
        print(f"{'Experiment':<30} {'Status':<12} {'Duration (s)':<15}")
        print("-" * 70)

        for name, result in self.experiment_results.items():
            status = result["status"]
            duration = result["duration"]
            icon = {
                "success": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "interrupted": "⏸️",
                "not_run": "⚪"
            }.get(status, "⚪")
            print(f"{icon} {name.replace('_', ' ').title():<27} {status.upper():<12} {duration:<15.1f}")
            if result.get("error"):
                print(f"   Error: {result['error'][:70]}...")

        print("-" * 70)

    def run_all(self, skip_baseline: bool, skip_ablations: bool, demo_only: bool) -> int:
        """
        Run all experiments in sequence.
        """
        try:
            if demo_only:
                print("\n📌 Running in DEMO-ONLY mode")
                self.run_demo()
            else:
                # Baseline
                if not skip_baseline:
                    self.run_baseline_comparison()
                else:
                    print("\n⏭️ Skipping baseline comparison (--skip-baseline)")
                    self.experiment_results["baseline_comparison"]["status"] = "skipped"

                # Ablations
                if not skip_ablations:
                    self.run_ablation_study()
                else:
                    print("\n⏭️ Skipping ablation study (--skip-ablations)")
                    self.experiment_results["ablation_study"]["status"] = "skipped"

                # Demo is optional in full pipeline; keep skipped unless user requests demo-only.
                print("\n⏭️ Skipping demo run (default). Use --demo-only to run demo.")
                self.experiment_results["demo_run"]["status"] = "skipped"

                # Governance stress test (standalone, does not affect benchmarks)
                try:
                    t0 = time.time()
                    run_governance_stress(self.gov_stress_dir, self.baseline_runner.base_config, num_examples=8)
                    self.experiment_results["governance_stress"]["status"] = "success"
                    self.experiment_results["governance_stress"]["duration"] = time.time() - t0
                except Exception as e:
                    self.experiment_results["governance_stress"]["status"] = "failed"
                    self.experiment_results["governance_stress"]["error"] = str(e)
                    print(f"❌ Governance stress test failed: {e}")

            # Print + persist orchestration summary
            self.generate_summary()
            self._write_experiment_summary()

            # Exit code: 0 only if all run (non-skipped) experiments succeeded
            for k, v in self.experiment_results.items():
                if v["status"] in ("failed", "interrupted"):
                    return 1
            return 0

        except KeyboardInterrupt:
            print("\n\n⚠️  Orchestration interrupted by user")
            self.experiment_results["baseline_comparison"]["status"] = (
                self.experiment_results["baseline_comparison"]["status"] if self.experiment_results["baseline_comparison"]["status"] != "running" else "interrupted"
            )
            self.experiment_results["ablation_study"]["status"] = (
                self.experiment_results["ablation_study"]["status"] if self.experiment_results["ablation_study"]["status"] != "running" else "interrupted"
            )
            self.experiment_results["demo_run"]["status"] = (
                self.experiment_results["demo_run"]["status"] if self.experiment_results["demo_run"]["status"] != "running" else "interrupted"
            )
            self.generate_summary()
            self._write_experiment_summary()
            return 130

        except Exception as e:
            print(f"\n\n❌ A fatal error occurred during orchestration: {e}")
            import traceback
            traceback.print_exc()
            self.generate_summary()
            self._write_experiment_summary()
            return 1


def main():
    """
    CLI entry.
    """
    parser = argparse.ArgumentParser(description="Run full SelfEvolve manuscript evaluation")
    parser.add_argument('--quick', action='store_true', help='Run a reduced quick test')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline comparison experiment (Table 1)')
    parser.add_argument('--skip-ablations', action='store_true', help='Skip ablation study experiment (Table 2)')
    parser.add_argument('--demo-only', action='store_true', help='Only run a single demo for qualitative analysis')
    args = parser.parse_args()

    runner = FullEvaluationRunner(quick_mode=args.quick)
    exit_code = runner.run_all(
        skip_baseline=args.skip_baseline,
        skip_ablations=args.skip_ablations,
        demo_only=args.demo_only
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
