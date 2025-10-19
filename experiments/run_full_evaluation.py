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

        # Track experiment results
        self.experiment_results = {
            "baseline_comparison": {"status": "not_run", "duration": 0, "error": None},
            "ablation_study": {"status": "not_run", "duration": 0, "error": None},
            "demo_run": {"status": "not_run", "duration": 0, "error": None}
        }

        # ✅ Initialize experiment objects once
        self.baseline_runner = BaselineComparison()
        self.ablation_runner = AblationStudy()

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
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_root}")
        print("=" * 70)
        print()

    def run_baseline_comparison(self) -> bool:
        """
        Run baseline comparison experiment (Table 1).
        """
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT 1: BASELINE COMPARISON (Table 1)")
        print("=" * 70)
        experiment_start = time.time()
        self.experiment_results["baseline_comparison"]["status"] = "running"

        try:
            print("🚀 Starting baseline comparison...")
            all_results = self.baseline_runner.run_all_experiments()
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

                # Demo is optional in non-demo-only mode → mark as skipped for clean exit code
                if self.experiment_results["demo_run"]["status"] == "not_run":
                    self.experiment_results["demo_run"]["status"] = "skipped"

            self.generate_summary()

            # Success if every tracked experiment either succeeded or was intentionally skipped
            if all(r["status"] in ["success", "skipped"] for r in self.experiment_results.values()):
                return 0  # Success
            else:
                return 1  # Failure

        except KeyboardInterrupt:
            print("\n\n⚠️ Evaluation interrupted by user")
            self.generate_summary()
            return 130

        except Exception as e:
            print(f"\n\n❌ A fatal error occurred during orchestration: {e}")
            import traceback
            traceback.print_exc()
            self.generate_summary()
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full evaluation for ACM manuscript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--quick', action='store_true', help='Run with reduced settings for fast testing')
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
