#!/usr/bin/env python3
"""
Quick data viewer for SELFEVOLVE results.

Usage:
    python scripts/view_results.py metrics       # View main metrics
    python scripts/view_results.py baseline      # View baseline comparison
    python scripts/view_results.py ablations     # View ablation study
    python scripts/view_results.py all           # View everything
    python scripts/view_results.py plot          # Generate plots
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def view_metrics():
    """View main evaluation metrics."""
    print_section("MAIN EVALUATION METRICS")

    metrics_path = Path("data/results/metrics.json")
    if not metrics_path.exists():
        print(f"❌ Not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})

    print(f"\n📊 Task Statistics:")
    print(f"   Total Tasks:        {summary.get('total_tasks', 0)}")
    print(f"   Successes:          {summary.get('successes', 0)}")
    print(f"   Failures:           {summary.get('failures', 0)}")
    print(f"   Success Rate:       {summary.get('success_rate', 0):.1%}")

    print(f"\n🔧 Patch Statistics:")
    print(f"   Proposed:           {summary.get('patches_proposed', 0)}")
    print(f"   Accepted:           {summary.get('patches_accepted', 0)}")
    print(f"   Acceptance Rate:    {summary.get('acceptance_rate', 0):.1%}")

    print(f"\n📈 Primary Metrics:")
    print(f"   RFR:                {summary.get('repeat_failure_rate', 0):.1%}")
    print(f"   CSR:                {summary.get('constraint_satisfaction_rate', 0):.1%}")
    print(f"   Rollback Freq:      {summary.get('rollback_frequency', 0):.1f}/1000")
    print(f"   Human Interventions: {summary.get('human_interventions', 0)}")

    print(f"\n⏱️ Time-to-Adapt (TTA):")
    tta = summary.get('time_to_adapt', {})
    if tta:
        for failure_class, tasks in tta.items():
            if tasks is not None:
                print(f"   {failure_class}: {tasks} tasks")
            else:
                print(f"   {failure_class}: ∞ (not adapted)")
    else:
        print("   No failures detected")

    print(f"\n🛡️ Governance Statistics:")
    gov = summary.get('governance_stats', {})
    print(f"   Value Checks:       {gov.get('value_checks', 0)} (vetoes: {gov.get('value_vetoes', 0)})")
    print(f"   Causal Checks:      {gov.get('causal_checks', 0)} (escalations: {gov.get('causal_escalations', 0)})")
    print(f"   Canary Tests:       {gov.get('canary_tests', 0)} (pass rate: {gov.get('canary_pass_rate', 0):.1%})")


def view_baseline_comparison():
    """View baseline comparison results."""
    print_section("BASELINE COMPARISON (Table 1)")

    csv_path = Path("experiments/results/baseline_comparison/table1_data.csv")
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        print("   Run: python experiments/run_baseline_comparison.py")
        return

    df = pd.read_csv(csv_path)

    # Group by system and compute statistics
    grouped = df.groupby('System').agg({
        'Success_Rate': ['mean', 'std'],
        'RFR': ['mean', 'std'],
        'TTA_Mean': 'mean',
        'Patches_Accepted': 'mean',
        'HI': 'sum'
    }).round(3)

    print("\n📊 Results by System:")
    print(grouped.to_string())

    print(f"\n💾 Full data: {csv_path}")


def view_ablations():
    """View ablation study results."""
    print_section("ABLATION STUDY (Table 2)")

    csv_path = Path("experiments/results/ablations/table2_data.csv")
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        print("   Run: python experiments/run_ablations.py")
        return

    df = pd.read_csv(csv_path)

    # Show key columns
    columns = ['Configuration', 'Success_Rate_Mean', 'RFR_Mean', 'TTA_Mean', 'Delta_TTA_vs_Full', 'Component_Removed']
    if all(col in df.columns for col in columns):
        print("\n📊 Component Contributions:")
        print(df[columns].to_string(index=False))
    else:
        print(df.to_string(index=False))

    print(f"\n💾 Full data: {csv_path}")


def view_per_failure():
    """View per-failure-class analysis."""
    print_section("PER-FAILURE-CLASS ANALYSIS (Table 3)")

    csv_path = Path("data/results/table3_per_failure.csv")
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        print("   Run: python main.py --mode demo")
        return

    df = pd.read_csv(csv_path)
    print("\n📊 Failure Class Breakdown:")
    print(df.to_string(index=False))

    print(f"\n💾 Full data: {csv_path}")


def view_governance():
    """View governance statistics."""
    print_section("GOVERNANCE STATISTICS (Table 4)")

    csv_path = Path("data/results/table4_governance.csv")
    if not csv_path.exists():
        print(f"❌ Not found: {csv_path}")
        print("   Run: python main.py --mode demo")
        return

    df = pd.read_csv(csv_path)
    print("\n📊 Governance Layer Performance:")
    print(df.to_string(index=False))

    print(f"\n💾 Full data: {csv_path}")


def view_summary():
    """View full evaluation summary."""
    print_section("FULL EVALUATION SUMMARY")

    summary_path = Path("experiments/results/evaluation_summary.json")
    if not summary_path.exists():
        print(f"❌ Not found: {summary_path}")
        print("   Run: python experiments/run_full_evaluation.py")
        return

    with open(summary_path) as f:
        data = json.load(f)

    print(f"\n⏰ Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"⚡ Quick Mode: {data.get('quick_mode', False)}")
    print(f"⏱️ Total Duration: {data.get('total_duration_seconds', 0):.1f}s")

    print(f"\n📋 Experiment Status:")
    for exp_name, result in data.get('experiments', {}).items():
        status = result.get('status', 'unknown')
        duration = result.get('duration', 0)

        icon = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
        exp_display = exp_name.replace("_", " ").title()
        print(f"   {icon} {exp_display:30} {status:12} ({duration:.1f}s)")

    print(f"\n📊 Tables Generated:")
    for table_name, table_path in data.get('tables_generated', {}).items():
        if table_path:
            print(f"   ✅ {table_name}: {table_path}")
        else:
            print(f"   ❌ {table_name}: NOT GENERATED")


def generate_plots():
    """Generate visualization plots."""
    print_section("GENERATING PLOTS")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        print("❌ matplotlib/seaborn not installed")
        print("   Install: pip install matplotlib seaborn")
        return

    output_dir = Path("experiments/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Baseline Comparison
    baseline_csv = Path("experiments/results/baseline_comparison/table1_data.csv")
    if baseline_csv.exists():
        df = pd.read_csv(baseline_csv)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Success Rate
        df.groupby('System')['Success_Rate'].mean().plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Success Rate by System')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_ylim([0, 1])

        # RFR
        df.groupby('System')['RFR'].mean().plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('Repeat Failure Rate (Lower is Better)')
        axes[1].set_ylabel('RFR')

        # TTA
        df.groupby('System')['TTA_Mean'].mean().plot(kind='bar', ax=axes[2], color='seagreen')
        axes[2].set_title('Time-to-Adapt (Lower is Better)')
        axes[2].set_ylabel('TTA (tasks)')

        plt.tight_layout()
        plot_path = output_dir / "baseline_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_path}")
        plt.close()

    # Plot 2: Ablation Study
    ablation_csv = Path("experiments/results/ablations/table2_data.csv")
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Component contribution to success rate
        df_plot = df.groupby('Component_Removed')['Success_Rate_Mean'].mean().sort_values()
        df_plot.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Component Contribution to Success Rate')
        ax.set_xlabel('Success Rate')
        ax.set_ylabel('Component Removed')

        plt.tight_layout()
        plot_path = output_dir / "ablation_study.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_path}")
        plt.close()

    print(f"\n📁 All plots saved to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "metrics":
        view_metrics()
    elif command == "baseline":
        view_baseline_comparison()
    elif command == "ablations":
        view_ablations()
    elif command == "failure":
        view_per_failure()
    elif command == "governance":
        view_governance()
    elif command == "summary":
        view_summary()
    elif command == "plot":
        generate_plots()
    elif command == "all":
        view_metrics()
        view_baseline_comparison()
        view_ablations()
        view_per_failure()
        view_governance()
        view_summary()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()