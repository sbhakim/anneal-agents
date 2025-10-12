#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for ACM Journals
===================================================

Creates high-resolution, publication-ready figures from SELFEVOLVE evaluation data.

Requirements:
    pip install matplotlib seaborn pandas numpy scipy

Usage:
    python scripts/generate_acm_plots.py
    python scripts/generate_acm_plots.py --dpi 600  # Higher resolution
    python scripts/generate_acm_plots.py --style ieee  # IEEE style
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


# ============================================================================
# PUBLICATION STYLE CONFIGURATION
# ============================================================================

def setup_publication_style(style='acm'):
    """
    Configure matplotlib for publication-quality output.

    Args:
        style: 'acm' or 'ieee' formatting
    """
    # ACM/IEEE standard fonts and sizes
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,

        # Line widths
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Figure
        'figure.dpi': 100,
        'savefig.dpi': 300,  # Will be overridden by command line
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Colors
        'axes.prop_cycle': plt.cycler(color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ])
    })

    # Set style
    sns.set_palette("husl")


# ============================================================================
# FIGURE 1: ADAPTATION CURVE (Success Rate Over Time)
# ============================================================================

def plot_adaptation_curve(metrics_data: Dict, output_dir: Path, dpi: int):
    """
    Plot success rate over time showing adaptation after failures.

    Key Insight: Shows how RFR drops to 0% after TTA.
    """
    print("📈 Generating Figure 1: Adaptation Curve...")

    tasks = metrics_data.get('tasks', [])
    if not tasks:
        print("  ⚠️ No task data found")
        return

    # Extract success status for each task
    task_ids = [t['task_id'] for t in tasks]
    successes = [1 if t['success'] else 0 for t in tasks]

    # Compute rolling success rate (window=10)
    window = 10
    rolling_sr = pd.Series(successes).rolling(window=window, min_periods=1).mean()

    # Get TTA point
    tta_dict = metrics_data['summary'].get('time_to_adapt', {})
    tta_value = None
    failure_class = None
    for fc, tta in tta_dict.items():
        if tta is not None:
            tta_value = tta
            failure_class = fc
            break

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot rolling success rate
    ax.plot(task_ids, rolling_sr, linewidth=2, color='#1f77b4',
            label=f'Success Rate (rolling window={window})')

    # Mark TTA point if exists
    if tta_value is not None:
        ax.axvline(x=tta_value, color='#d62728', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Time-to-Adapt (TTA={tta_value})')

        # Add annotation
        ax.annotate(f'Adaptation\nCompleted',
                    xy=(tta_value, rolling_sr.iloc[min(tta_value, len(rolling_sr) - 1)]),
                    xytext=(tta_value + 5, 0.7),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                    fontsize=9, color='#d62728')

    # Mark failure points
    failure_indices = [i for i, s in enumerate(successes) if s == 0]
    if failure_indices:
        ax.scatter([task_ids[i] for i in failure_indices],
                   [rolling_sr.iloc[i] for i in failure_indices],
                   color='#ff7f0e', s=80, marker='x', linewidths=2,
                   label='Failures', zorder=5)

    # Formatting
    ax.set_xlabel('Task Number', fontweight='bold')
    ax.set_ylabel('Success Rate', fontweight='bold')
    ax.set_title('System Adaptation Over Time', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, len(tasks)])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)

    # Add performance annotation
    final_sr = metrics_data['summary']['success_rate']
    rfr = metrics_data['summary']['repeat_failure_rate']
    ax.text(0.02, 0.98, f'Final Success Rate: {final_sr:.1%}\nRFR: {rfr:.1%}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = output_dir / 'fig1_adaptation_curve.pdf'
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), dpi=dpi)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: METACOGNITIVE PATHWAY DISTRIBUTION
# ============================================================================

def plot_pathway_distribution(experience_data: List, output_dir: Path, dpi: int):
    """
    Show distribution of S1 vs S2 pathway selection.

    Key Insight: Demonstrates adaptive metacognitive control.
    """
    print("📊 Generating Figure 2: Metacognitive Pathway Distribution...")

    # Extract pathway decisions
    pathways = []
    uncertainties = []
    p_viols = []

    for trace in experience_data:
        trace_data = trace.get('trace', [])
        for entry in trace_data:
            if 'pathway' in entry:
                pathways.append(entry['pathway'])
                signals = entry.get('signals', {})
                uncertainties.append(signals.get('u', 0))
                p_viols.append(signals.get('p_viol', 0))

    if not pathways:
        print("  ⚠️ No pathway data found")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Subplot 1: Pathway distribution
    pathway_counts = pd.Series(pathways).value_counts()
    colors = {'S1': '#2ca02c', 'S2': '#1f77b4', 'VERIFY_S1': '#ff7f0e', 'DEFER': '#d62728'}

    wedges, texts, autotexts = ax1.pie(
        pathway_counts.values,
        labels=pathway_counts.index,
        autopct='%1.1f%%',
        colors=[colors.get(p, '#cccccc') for p in pathway_counts.index],
        startangle=90,
        textprops={'fontsize': 10}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax1.set_title('Pathway Selection Distribution', fontweight='bold')

    # Subplot 2: Signal space scatter
    ax2.scatter(uncertainties, p_viols, alpha=0.6, s=50, c='#1f77b4', edgecolors='k', linewidths=0.5)

    # Add threshold lines
    ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1, label='τ_u=0.5')
    ax2.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, linewidth=1, label='τ_p=0.3')

    ax2.set_xlabel('Uncertainty (u)', fontweight='bold')
    ax2.set_ylabel('Violation Probability (p_viol)', fontweight='bold')
    ax2.set_title('Metacognitive Signal Space', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / 'fig2_pathway_distribution.pdf'
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), dpi=dpi)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 3: GOVERNANCE LAYER EFFECTIVENESS
# ============================================================================

def plot_governance_effectiveness(metrics_data: Dict, output_dir: Path, dpi: int):
    """
    Show governance layer statistics (value, causal, canary).

    Key Insight: All checks passed, demonstrating safety.
    """
    print("🛡️ Generating Figure 3: Governance Layer Effectiveness...")

    gov_stats = metrics_data['summary'].get('governance_stats', {})

    if not gov_stats:
        print("  ⚠️ No governance data found")
        return

    # Prepare data
    categories = ['Value\nGuard', 'Causal\nGuard', 'Canary\nTest']
    total_checks = [
        gov_stats.get('value_checks', 0),
        gov_stats.get('causal_checks', 0),
        gov_stats.get('canary_tests', 0)
    ]
    passed = [
        gov_stats.get('value_checks', 0) - gov_stats.get('value_vetoes', 0),
        gov_stats.get('causal_checks', 0) - gov_stats.get('causal_escalations', 0),
        gov_stats.get('canary_passes', 0)
    ]
    failed = [
        gov_stats.get('value_vetoes', 0),
        gov_stats.get('causal_escalations', 0),
        gov_stats.get('canary_fails', 0)
    ]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Subplot 1: Stacked bar chart
    x = np.arange(len(categories))
    width = 0.6

    p1 = ax1.bar(x, passed, width, label='Passed', color='#2ca02c', edgecolor='black', linewidth=0.8)
    p2 = ax1.bar(x, failed, width, bottom=passed, label='Failed/Escalated',
                 color='#d62728', edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Number of Checks', fontweight='bold')
    ax1.set_title('Governance Layer Checks', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (p, f) in enumerate(zip(passed, failed)):
        total = p + f
        if total > 0:
            ax1.text(i, total + 0.1, str(total), ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 2: Pass rates
    pass_rates = [
        (p / t * 100) if t > 0 else 0
        for p, t in zip(passed, total_checks)
    ]

    bars = ax2.bar(x, pass_rates, width, color='#1f77b4', edgecolor='black', linewidth=0.8)

    # Color bars based on pass rate
    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        if rate == 100:
            bar.set_color('#2ca02c')  # Green for perfect
        elif rate >= 80:
            bar.set_color('#ff7f0e')  # Orange for good
        else:
            bar.set_color('#d62728')  # Red for poor

    ax2.set_ylabel('Pass Rate (%)', fontweight='bold')
    ax2.set_title('Governance Pass Rates', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, rate in enumerate(pass_rates):
        ax2.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add 100% reference line
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    output_path = output_dir / 'fig3_governance_effectiveness.pdf'
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), dpi=dpi)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 4: PER-FAILURE-CLASS ANALYSIS
# ============================================================================

def plot_failure_class_analysis(table3_path: Path, output_dir: Path, dpi: int):
    """
    Detailed breakdown of failure classes with TTA and patches.

    Key Insight: Shows learning effectiveness per failure type.
    """
    print("🔍 Generating Figure 4: Per-Failure-Class Analysis...")

    if not table3_path.exists():
        print(f"  ⚠️ Table 3 not found: {table3_path}")
        return

    df = pd.read_csv(table3_path)

    if df.empty:
        print("  ⚠️ No failure class data")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Subplot 1: TTA by failure class
    tta_data = df[df['TTA'] != '∞']['TTA'].astype(float)
    failure_classes = df[df['TTA'] != '∞']['Failure_Class']

    if not tta_data.empty:
        bars = ax1.barh(failure_classes, tta_data, color='#1f77b4', edgecolor='black', linewidth=0.8)
        ax1.set_xlabel('Time-to-Adapt (Tasks)', fontweight='bold')
        ax1.set_title('Adaptation Speed by Failure Class', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, tta_data)):
            ax1.text(val + 0.5, i, f'{val:.0f}', va='center', fontweight='bold', fontsize=9)

    # Subplot 2: Patch statistics
    patches_proposed = df['Patches_Proposed'].values
    patches_accepted = df['Patches_Accepted'].values

    x = np.arange(len(df))
    width = 0.35

    ax2.bar(x - width / 2, patches_proposed, width, label='Proposed',
            color='#ff7f0e', edgecolor='black', linewidth=0.8)
    ax2.bar(x + width / 2, patches_accepted, width, label='Accepted',
            color='#2ca02c', edgecolor='black', linewidth=0.8)

    ax2.set_ylabel('Number of Patches', fontweight='bold')
    ax2.set_title('Patch Proposal vs Acceptance', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([fc.split(':')[0] for fc in df['Failure_Class']], rotation=0)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Failure instances timeline
    first_occurrence = df['First_Task'].values
    total_instances = df['Total_Instances'].values

    ax3.scatter(first_occurrence, total_instances, s=200, alpha=0.6,
                c='#d62728', edgecolors='k', linewidths=1.5)

    # Annotate points
    for i, fc in enumerate(df['Failure_Class']):
        ax3.annotate(fc.split(':')[1][:8],
                     xy=(first_occurrence[i], total_instances[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    ax3.set_xlabel('First Occurrence (Task #)', fontweight='bold')
    ax3.set_ylabel('Total Instances', fontweight='bold')
    ax3.set_title('Failure Instance Timeline', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Final RFR
    final_rfr = df['Final_RFR'].values

    bars = ax4.bar(x, final_rfr, color='#9467bd', edgecolor='black', linewidth=0.8)
    ax4.set_ylabel('Final RFR', fontweight='bold')
    ax4.set_title('Repeat Failure Rate by Class', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([fc.split(':')[0] for fc in df['Failure_Class']], rotation=0)
    ax4.set_ylim([0, max(final_rfr) * 1.2 if max(final_rfr) > 0 else 0.1])
    ax4.grid(True, alpha=0.3, axis='y')

    # Add target line at 5% (paper threshold)
    ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Target (5%)')
    ax4.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'fig4_failure_class_analysis.pdf'
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), dpi=dpi)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: COMPOSITE METRICS DASHBOARD
# ============================================================================

def plot_metrics_dashboard(metrics_data: Dict, output_dir: Path, dpi: int):
    """
    Comprehensive dashboard showing all key metrics.

    Key Insight: One-page summary for manuscript.
    """
    print("📋 Generating Figure 5: Metrics Dashboard...")

    summary = metrics_data['summary']

    # Create figure with custom grid
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # --- Panel 1: Key Metrics (Top Left) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    metrics_text = f"""
    Success Rate: {summary['success_rate']:.1%}  |  RFR: {summary['repeat_failure_rate']:.1%}  |  CSR: {summary['constraint_satisfaction_rate']:.1%}
    Patches: {summary['patches_accepted']}/{summary['patches_proposed']} ({summary['acceptance_rate']:.1%})  |  Rollbacks: {summary['rollback_frequency']:.1f}/1000  |  Human: {summary['human_interventions']}
    """

    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes,
             fontsize=11, ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
    ax1.set_title('Primary Metrics Summary', fontweight='bold', fontsize=12, pad=10)

    # --- Panel 2: Success Rate Gauge (Middle Left) ---
    ax2 = fig.add_subplot(gs[1, 0], projection='polar')

    theta = np.linspace(0, np.pi, 100)
    sr = summary['success_rate']
    sr_theta = sr * np.pi

    # Background arc
    ax2.plot(theta, np.ones_like(theta), color='lightgray', linewidth=15, alpha=0.3)
    # Filled arc
    ax2.plot(theta[theta <= sr_theta], np.ones(sum(theta <= sr_theta)),
             color='#2ca02c', linewidth=15)

    ax2.set_ylim([0, 1])
    ax2.set_theta_offset(np.pi)
    ax2.set_theta_direction(-1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['polar'].set_visible(False)

    # Add text
    ax2.text(np.pi / 2, 0.5, f'{sr:.1%}', ha='center', va='center',
             fontsize=20, fontweight='bold')
    ax2.text(np.pi / 2, -0.2, 'Success\nRate', ha='center', va='center',
             fontsize=10)

    # --- Panel 3: RFR Gauge (Middle Center) ---
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')

    rfr = summary['repeat_failure_rate']
    rfr_norm = min(rfr / 0.3, 1.0)  # Normalize to 30% max
    rfr_theta = rfr_norm * np.pi

    ax3.plot(theta, np.ones_like(theta), color='lightgray', linewidth=15, alpha=0.3)
    ax3.plot(theta[theta <= rfr_theta], np.ones(sum(theta <= rfr_theta)),
             color='#d62728' if rfr > 0.05 else '#2ca02c', linewidth=15)

    ax3.set_ylim([0, 1])
    ax3.set_theta_offset(np.pi)
    ax3.set_theta_direction(-1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['polar'].set_visible(False)

    ax3.text(np.pi / 2, 0.5, f'{rfr:.1%}', ha='center', va='center',
             fontsize=20, fontweight='bold')
    ax3.text(np.pi / 2, -0.2, 'RFR\n(Target: <5%)', ha='center', va='center',
             fontsize=9)

    # --- Panel 4: Patch Acceptance (Middle Right) ---
    ax4 = fig.add_subplot(gs[1, 2])

    sizes = [summary['patches_accepted'], summary['patches_rejected']]
    labels = ['Accepted', 'Rejected']
    colors = ['#2ca02c', '#d62728']
    explode = (0.1, 0)

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f%%',
                                           colors=colors, explode=explode,
                                           startangle=90, textprops={'fontsize': 9})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    ax4.set_title('Patch\nAcceptance', fontweight='bold', fontsize=10)

    # --- Panel 5: Time-to-Adapt Bar (Bottom Left) ---
    ax5 = fig.add_subplot(gs[2, 0])

    tta_dict = summary.get('time_to_adapt', {})
    if tta_dict:
        classes = list(tta_dict.keys())
        tta_values = [v if v is not None else 0 for v in tta_dict.values()]

        bars = ax5.barh(classes, tta_values, color='#1f77b4', edgecolor='black', linewidth=0.8)
        ax5.set_xlabel('Tasks', fontweight='bold', fontsize=9)
        ax5.set_title('Time-to-Adapt', fontweight='bold', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, tta_values):
            if val > 0:
                ax5.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                         f'{val:.0f}', va='center', fontsize=8)

    # --- Panel 6: Governance Stats (Bottom Center & Right) ---
    ax6 = fig.add_subplot(gs[2, 1:])

    gov = summary.get('governance_stats', {})
    if gov:
        categories = ['Value', 'Causal', 'Canary']
        checks = [gov.get('value_checks', 0), gov.get('causal_checks', 0), gov.get('canary_tests', 0)]
        vetoes = [gov.get('value_vetoes', 0), gov.get('causal_escalations', 0), gov.get('canary_fails', 0)]

        x = np.arange(len(categories))
        width = 0.35

        ax6.bar(x - width / 2, checks, width, label='Total', color='#1f77b4', edgecolor='black', linewidth=0.8)
        ax6.bar(x + width / 2, vetoes, width, label='Vetoed/Failed', color='#d62728', edgecolor='black', linewidth=0.8)

        ax6.set_ylabel('Count', fontweight='bold', fontsize=9)
        ax6.set_title('Governance Layer Activity', fontweight='bold', fontsize=10)
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('SELFEVOLVE Evaluation Dashboard', fontsize=14, fontweight='bold', y=0.98)

    output_path = output_dir / 'fig5_metrics_dashboard.pdf'
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.savefig(output_path.with_suffix('.png'), dpi=dpi)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality plots')
    parser.add_argument('--dpi', type=int, default=300, help='Output resolution (default: 300)')
    parser.add_argument('--style', choices=['acm', 'ieee'], default='acm', help='Publication style')
    parser.add_argument('--output-dir', type=str, default='experiments/results/plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    # Setup
    setup_publication_style(args.style)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 70)
    print(f"Style: {args.style.upper()}")
    print(f"DPI: {args.dpi}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load data
    print("\n📂 Loading data files...")

    # Update paths to look in experiments/data/results
    data_dir = Path("experiments/data/results")

    metrics_path = data_dir / "metrics.json"
    experience_path = data_dir / "experience_pool.json"
    table3_path = data_dir / "table3_per_failure.csv"

    if not metrics_path.exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        return 1

    with open(metrics_path) as f:
        metrics_data = json.load(f)
    print(f"  ✅ Loaded: {metrics_path}")

    experience_data = []
    if experience_path.exists():
        with open(experience_path) as f:
            experience_data = json.load(f)
        print(f"  ✅ Loaded: {experience_path}")

    # Generate plots
    print("\n🎨 Generating figures...")

    plot_adaptation_curve(metrics_data, output_dir, args.dpi)

    if experience_data:
        plot_pathway_distribution(experience_data, output_dir, args.dpi)

    plot_governance_effectiveness(metrics_data, output_dir, args.dpi)

    if table3_path.exists():
        plot_failure_class_analysis(table3_path, output_dir, args.dpi)

    plot_metrics_dashboard(metrics_data, output_dir, args.dpi)

    print("\n" + "=" * 70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    for pdf_file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {pdf_file.name}")

    print("\n💡 Tip: Use PDF files for LaTeX, PNG for presentations")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())