#!/usr/bin/env python3
# experiments/run_full_evaluation.py
"""
Master Evaluation Script for ACM Manuscript
============================================

Runs all experiments required for manuscript data generation:
- Baseline Comparison → Table 1 (Main Results)
- Ablation Study → Table 2 (Component Contributions)
- Demo Run → Tables 3-4 (Per-Failure Analysis & Governance)

Usage:
    python experiments/run_full_evaluation.py [--quick] [--skip-baseline] [--skip-ablations]

Options:
    --quick           Run with reduced settings (5 tasks, 2 seeds) for testing
    --skip-baseline   Skip baseline comparison experiment
    --skip-ablations  Skip ablation study experiment
    --demo-only       Run only demo for Tables 3-4
    --help            Show this help message

Author: SELFEVOLVE Team
Date: 2025-01-12
"""

import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config


class FullEvaluationRunner:
    """
    Orchestrates all experiments for manuscript data generation.
    """

    def __init__(self, quick_mode: bool = False):
        """
        Initialize evaluation runner.

        Args:
            quick_mode: If True, use reduced settings for testing
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

        # Ensure directories exist
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

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT 1: BASELINE COMPARISON (Table 1)")
        print("=" * 70)
        print("Purpose: Compare SELFEVOLVE against 3 baselines")
        print("Output: experiments/results/baseline_comparison/table1_data.csv")
        print()

        if self.quick_mode:
            print("⚡ QUICK MODE: Running with reduced settings")
            print("   - 10 tasks (normally 50)")
            print("   - 2 seeds (normally 5)")
            print()

        experiment_start = time.time()
        self.experiment_results["baseline_comparison"]["status"] = "running"

        try:
            # Import and run directly for better control
            from experiments.run_baseline_comparison import BaselineComparison

            # Modify config if quick mode
            if self.quick_mode:
                comparison = BaselineComparison()
                comparison.num_tasks = 10
                comparison.num_seeds = 2
                comparison.seeds = [42, 123]
            else:
                comparison = BaselineComparison()

            print("🚀 Starting baseline comparison...")
            all_results = comparison.run_all_experiments()
            comparison.print_summary(all_results)

            # Check if table1_data.csv was generated
            csv_path = self.results_root / "baseline_comparison" / "table1_data.csv"
            if csv_path.exists():
                print(f"\n✅ Table 1 data generated: {csv_path}")
                self.experiment_results["baseline_comparison"]["status"] = "success"
            else:
                print(f"\n⚠️ Warning: Table 1 CSV not found at {csv_path}")
                self.experiment_results["baseline_comparison"]["status"] = "partial"

            duration = time.time() - experiment_start
            self.experiment_results["baseline_comparison"]["duration"] = duration

            print(f"\n✅ Baseline comparison completed in {duration:.1f}s")
            return True

        except KeyboardInterrupt:
            print("\n⚠️ Baseline comparison interrupted by user")
            self.experiment_results["baseline_comparison"]["status"] = "interrupted"
            raise

        except Exception as e:
            print(f"\n❌ Baseline comparison failed: {e}")
            import traceback
            traceback.print_exc()

            self.experiment_results["baseline_comparison"]["status"] = "failed"
            self.experiment_results["baseline_comparison"]["error"] = str(e)

            return False

    def run_ablation_study(self) -> bool:
        """
        Run ablation study (Table 2).

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("🔬 EXPERIMENT 2: ABLATION STUDY (Table 2)")
        print("=" * 70)
        print("Purpose: Measure contribution of each component")
        print("Output: experiments/results/ablations/table2_data.csv")
        print()

        if self.quick_mode:
            print("⚡ QUICK MODE: Running with reduced settings")
            print("   - 10 tasks (normally 50)")
            print("   - 2 seeds (normally 3)")
            print()

        experiment_start = time.time()
        self.experiment_results["ablation_study"]["status"] = "running"

        try:
            # Import and run directly
            from experiments.run_ablations import AblationStudy

            # Modify config if quick mode
            if self.quick_mode:
                study = AblationStudy()
                study.num_tasks = 10
                study.num_seeds = 2
                study.seeds = [42, 123]
            else:
                study = AblationStudy()

            print("🚀 Starting ablation study...")
            all_results = study.run_all_ablations()
            study.print_summary(all_results)

            # Check if table2_data.csv was generated
            csv_path = self.results_root / "ablations" / "table2_data.csv"
            if csv_path.exists():
                print(f"\n✅ Table 2 data generated: {csv_path}")
                self.experiment_results["ablation_study"]["status"] = "success"
            else:
                print(f"\n⚠️ Warning: Table 2 CSV not found at {csv_path}")
                self.experiment_results["ablation_study"]["status"] = "partial"

            duration = time.time() - experiment_start
            self.experiment_results["ablation_study"]["duration"] = duration

            print(f"\n✅ Ablation study completed in {duration:.1f}s")
            return True

        except KeyboardInterrupt:
            print("\n⚠️ Ablation study interrupted by user")
            self.experiment_results["ablation_study"]["status"] = "interrupted"
            raise

        except Exception as e:
            print(f"\n❌ Ablation study failed: {e}")
            import traceback
            traceback.print_exc()

            self.experiment_results["ablation_study"]["status"] = "failed"
            self.experiment_results["ablation_study"]["error"] = str(e)

            return False

    def run_demo(self) -> bool:
        """
        Run demo mode for detailed per-failure analysis (Tables 3-4).

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("🎯 EXPERIMENT 3: DEMO RUN (Tables 3-4)")
        print("=" * 70)
        print("Purpose: Generate per-failure-class and governance data")
        print("Output: data/results/table3_per_failure.csv")
        print("        data/results/table4_governance.csv")
        print()

        if self.quick_mode:
            print("⚡ QUICK MODE: Running with reduced settings")
            print("   - 10 tasks (normally 20)")
            print()

        experiment_start = time.time()
        self.experiment_results["demo_run"]["status"] = "running"

        try:
            # Import and run directly
            from src.core.system import SelfEvolveSystem

            # Load config
            config = load_config("config.yaml")

            # Override for demo mode
            config['logging']['level'] = 'INFO'

            if self.quick_mode:
                config['scenario']['num_tasks'] = 10
            else:
                config['scenario']['num_tasks'] = 20

            # Apply demo overrides
            demo_overrides = config.get('demo', {})
            config['scenario']['failure_rate'] = demo_overrides.get('failure_rate', 0.4)
            config['scenario']['min_failures_in_prefix'] = demo_overrides.get('min_failures_in_prefix', 5)

            print("🚀 Starting demo run...")
            system = SelfEvolveSystem(config)
            metrics = system.run_evaluation()

            # Check if tables were generated
            table3_path = self.data_root / "table3_per_failure.csv"
            table4_path = self.data_root / "table4_governance.csv"

            success = True
            if table3_path.exists():
                print(f"\n✅ Table 3 data generated: {table3_path}")
            else:
                print(f"\n⚠️ Warning: Table 3 CSV not found at {table3_path}")
                success = False

            if table4_path.exists():
                print(f"✅ Table 4 data generated: {table4_path}")
            else:
                print(f"⚠️ Warning: Table 4 CSV not found at {table4_path}")
                success = False

            if success:
                self.experiment_results["demo_run"]["status"] = "success"
            else:
                self.experiment_results["demo_run"]["status"] = "partial"

            duration = time.time() - experiment_start
            self.experiment_results["demo_run"]["duration"] = duration

            print(f"\n✅ Demo run completed in {duration:.1f}s")
            return success

        except KeyboardInterrupt:
            print("\n⚠️ Demo run interrupted by user")
            self.experiment_results["demo_run"]["status"] = "interrupted"
            raise

        except Exception as e:
            print(f"\n❌ Demo run failed: {e}")
            import traceback
            traceback.print_exc()

            self.experiment_results["demo_run"]["status"] = "failed"
            self.experiment_results["demo_run"]["error"] = str(e)

            return False

    def generate_summary(self) -> None:
        """
        Generate comprehensive summary of all experiments.
        """
        print("\n" + "=" * 70)
        print("📋 EVALUATION SUMMARY")
        print("=" * 70)

        total_duration = time.time() - self.start_time

        print(f"\nTotal Elapsed Time: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
        print()

        # Print results for each experiment
        print("Experiment Results:")
        print("-" * 70)

        for exp_name, result in self.experiment_results.items():
            status = result["status"]
            duration = result["duration"]

            # Status icon
            if status == "success":
                icon = "✅"
            elif status == "partial":
                icon = "⚠️"
            elif status == "failed":
                icon = "❌"
            elif status == "interrupted":
                icon = "⏸️"
            else:
                icon = "⏭️"

            exp_display = exp_name.replace("_", " ").title()
            print(f"{icon} {exp_display:30} {status:12} ({duration:.1f}s)")

            if result["error"]:
                print(f"   Error: {result['error'][:60]}...")

        print("-" * 70)

        # Check which tables were generated
        print("\nGenerated Data Tables:")
        print("-" * 70)

        tables = [
            ("Table 1 (Baselines)", self.results_root / "baseline_comparison" / "table1_data.csv"),
            ("Table 2 (Ablations)", self.results_root / "ablations" / "table2_data.csv"),
            ("Table 3 (Per-Failure)", self.data_root / "table3_per_failure.csv"),
            ("Table 4 (Governance)", self.data_root / "table4_governance.csv"),
        ]

        for table_name, table_path in tables:
            if table_path.exists():
                size_kb = table_path.stat().st_size / 1024
                print(f"✅ {table_name:25} {table_path} ({size_kb:.1f} KB)")
            else:
                print(f"❌ {table_name:25} NOT FOUND")

        print("-" * 70)

        # Overall status
        success_count = sum(1 for r in self.experiment_results.values() if r["status"] == "success")
        total_count = len(self.experiment_results)

        print(f"\nOverall: {success_count}/{total_count} experiments completed successfully")

        if success_count == total_count:
            print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("   Ready for manuscript submission.")
        elif success_count > 0:
            print("\n⚠️ PARTIAL SUCCESS")
            print("   Some experiments completed. Review failures above.")
        else:
            print("\n❌ EVALUATION FAILED")
            print("   No experiments completed successfully.")

        print("\n" + "=" * 70)

        # Save summary to JSON
        summary_path = self.results_root / "evaluation_summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "quick_mode": self.quick_mode,
            "experiments": self.experiment_results,
            "tables_generated": {
                table_name: str(table_path) if table_path.exists() else None
                for table_name, table_path in tables
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"📄 Summary saved to: {summary_path}")
        print("=" * 70)

    def run_all(self, skip_baseline: bool = False, skip_ablations: bool = False,
                demo_only: bool = False) -> int:
        """
        Run all experiments in sequence.

        Args:
            skip_baseline: If True, skip baseline comparison
            skip_ablations: If True, skip ablation study
            demo_only: If True, only run demo

        Returns:
            0 if all successful, 1 if any failed
        """
        try:
            if demo_only:
                print("\n📌 Running in DEMO-ONLY mode")
                self.run_demo()
            else:
                # Run experiments in order
                if not skip_baseline:
                    success = self.run_baseline_comparison()
                    if not success:
                        print("\n⚠️ Baseline comparison failed. Continuing with remaining experiments...")
                else:
                    print("\n⏭️ Skipping baseline comparison (--skip-baseline)")
                    self.experiment_results["baseline_comparison"]["status"] = "skipped"

                if not skip_ablations:
                    success = self.run_ablation_study()
                    if not success:
                        print("\n⚠️ Ablation study failed. Continuing with remaining experiments...")
                else:
                    print("\n⏭️ Skipping ablation study (--skip-ablations)")
                    self.experiment_results["ablation_study"]["status"] = "skipped"

                # Always run demo for Tables 3-4
                self.run_demo()

            # Generate summary
            self.generate_summary()

            # Determine exit code
            if all(r["status"] in ["success", "skipped"] for r in self.experiment_results.values()):
                return 0
            else:
                return 1

        except KeyboardInterrupt:
            print("\n\n⚠️ Evaluation interrupted by user")
            self.generate_summary()
            return 130  # Standard exit code for SIGINT

        except Exception as e:
            print(f"\n\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            self.generate_summary()
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full evaluation for ACM manuscript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (recommended for manuscript)
  python experiments/run_full_evaluation.py

  # Quick test run (for debugging)
  python experiments/run_full_evaluation.py --quick

  # Only generate Tables 3-4
  python experiments/run_full_evaluation.py --demo-only

  # Skip expensive baseline comparison
  python experiments/run_full_evaluation.py --skip-baseline
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run with reduced settings (10 tasks, 2 seeds) for testing'
    )

    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline comparison experiment (Table 1)'
    )

    parser.add_argument(
        '--skip-ablations',
        action='store_true',
        help='Skip ablation study experiment (Table 2)'
    )

    parser.add_argument(
        '--demo-only',
        action='store_true',
        help='Only run demo for Tables 3-4 (fastest option)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.demo_only and (args.skip_baseline or args.skip_ablations):
        print("⚠️ Warning: --demo-only overrides --skip-* flags")

    # Create runner and execute
    runner = FullEvaluationRunner(quick_mode=args.quick)
    exit_code = runner.run_all(
        skip_baseline=args.skip_baseline,
        skip_ablations=args.skip_ablations,
        demo_only=args.demo_only
    )

    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()