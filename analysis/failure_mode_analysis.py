from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path
import json
import csv
import numpy as np
from datetime import datetime


class FailureModeAnalyzer:
    """
    Analyzes repeat failure rate (RFR) to identify root causes.
    Addresses the 84% observed RFR reported in manuscript Section XI.
    """

    def __init__(self, output_dir: str = "Results/failure_modes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.categories = {
            'localization_error': 'Wrong operator identified as responsible',
            'incomplete_patch': 'Patch addresses symptom, not root cause',
            'scoring_error': 'Low-quality patch scored too high',
            'guardrail_rejection': 'Valid patch vetoed by guardrails',
            'canary_failure': 'Patch fails sandbox testing',
            'multiple_faults': 'Multiple simultaneous failures',
        }

    def analyze_rfr_sources(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Categorize repeat failures by root cause.
        Returns breakdown and examples.
        """

        category_counts = defaultdict(int)
        category_examples = defaultdict(list)
        total_repeats = 0
        repeat_distribution = Counter()

        for trace in traces:
            repeat_count = trace.get('repeat_count', 1)

            if repeat_count > 1:
                total_repeats += 1
                repeat_distribution[repeat_count] += 1

                category = self._classify_failure_cause(trace)
                category_counts[category] += 1

                if len(category_examples[category]) < 3:  # Keep top 3 examples
                    category_examples[category].append({
                        'task_id': trace.get('task_id'),
                        'operator': trace.get('operator'),
                        'repeat_count': repeat_count,
                        'failure_type': trace.get('error_type'),
                        'patches_attempted': len(trace.get('patch_attempts', [])),
                    })

        # Calculate percentages
        breakdown = {}
        if total_repeats > 0:
            for category, count in category_counts.items():
                breakdown[category] = {
                    'count': count,
                    'percentage': (count / total_repeats) * 100,
                    'description': self.categories.get(category, 'Unknown'),
                    'examples': category_examples[category],
                }

        return {
            'total_repeat_failures': total_repeats,
            'breakdown': breakdown,
            'repeat_distribution': dict(repeat_distribution),
            'max_repeats_observed': max(repeat_distribution.keys()) if repeat_distribution else 0,
        }

    def _classify_failure_cause(self, trace: Dict[str, Any]) -> str:
        """Classify what caused the repeat failure."""

        patch_attempts = trace.get('patch_attempts', [])
        if not patch_attempts:
            return 'incomplete_patch'

        last_patch = patch_attempts[-1]

        # Check localization accuracy
        if self._is_localization_error(trace, last_patch):
            return 'localization_error'

        # Check scoring quality
        if self._is_scoring_error(last_patch):
            return 'scoring_error'

        # Check guardrail behavior
        if last_patch.get('guardrail_verdict') == 'VETO':
            return 'guardrail_rejection'

        # Check canary results
        if last_patch.get('canary_result', {}).get('verdict') == 'FAIL':
            return 'canary_failure'

        # Check for incomplete fixes
        if self._is_incomplete_patch(trace, last_patch):
            return 'incomplete_patch'

        # Multiple concurrent issues
        if len(trace.get('active_constraints', [])) > 1:
            return 'multiple_faults'

        return 'incomplete_patch'  # Default

    def _is_localization_error(self, trace: Dict[str, Any], patch: Dict[str, Any]) -> bool:
        """Check if wrong operator was blamed."""

        actual_failing_op = trace.get('actual_failing_operator')
        patched_op = patch.get('target_operator')

        if actual_failing_op and patched_op:
            return actual_failing_op != patched_op

        # Heuristic: if patch had no effect on failure rate
        if patch.get('effectiveness_score', 1.0) < 0.1:
            return True

        return False

    def _is_scoring_error(self, patch: Dict[str, Any]) -> bool:
        """Check if low-quality patch scored too high."""

        scores = patch.get('scores', {})
        aggregate = scores.get('aggregate', 0)

        # High aggregate but failed in practice
        if aggregate > 0.7 and patch.get('outcome') == 'failed':
            return True

        # Individual dimension mismatches
        plausibility = scores.get('plausibility', 0)
        consistency = scores.get('consistency', 1)

        if plausibility < 0.3 and aggregate > 0.5:  # Implausible but scored well
            return True

        if consistency < 0.5:  # Logically inconsistent
            return True

        return False

    def _is_incomplete_patch(self, trace: Dict[str, Any], patch: Dict[str, Any]) -> bool:
        """Check if patch only addressed surface issue."""

        # Patch adds precondition but doesn't fix underlying issue
        if patch.get('action') == 'ADD_PRECONDITION':
            if trace.get('repeat_count', 1) > 2:  # Still failing after precondition
                return True

        # Patch scope too narrow
        affected_params = patch.get('affected_parameters', [])
        if len(affected_params) < 2 and trace.get('repeat_count', 1) > 1:
            return True

        return False

    def compute_learning_curve(self, traces: List[Dict[str, Any]],
                              window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze how RFR changes over time.
        Tests if system learns to reduce repeat failures.
        """

        if not traces:
            return {'windows': [], 'rfr_over_time': [], 'trend': 'insufficient_data'}

        sorted_traces = sorted(traces, key=lambda t: t.get('task_id', 0))

        windows = []
        rfr_over_time = []
        success_rate_over_time = []

        for i in range(0, len(sorted_traces), window_size):
            window = sorted_traces[i:i + window_size]

            if len(window) < window_size // 2:  # Skip incomplete windows
                continue

            window_rfr = self._compute_window_rfr(window)
            window_sr = self._compute_window_success_rate(window)

            windows.append(i // window_size)
            rfr_over_time.append(window_rfr)
            success_rate_over_time.append(window_sr)

        # Compute trend
        if len(rfr_over_time) >= 3:
            trend = self._compute_trend(rfr_over_time)
        else:
            trend = 'insufficient_data'

        return {
            'windows': windows,
            'rfr_over_time': rfr_over_time,
            'success_rate_over_time': success_rate_over_time,
            'trend': trend,
            'initial_rfr': rfr_over_time[0] if rfr_over_time else 0,
            'final_rfr': rfr_over_time[-1] if rfr_over_time else 0,
            'improvement': (rfr_over_time[0] - rfr_over_time[-1]) if len(rfr_over_time) >= 2 else 0,
        }

    def _compute_window_rfr(self, window: List[Dict[str, Any]]) -> float:
        """Compute RFR for a window of tasks."""

        failures_in_window = [t for t in window if t.get('status') == 'failure']
        if not failures_in_window:
            return 0.0

        repeat_failures = [t for t in failures_in_window if t.get('repeat_count', 1) > 1]

        return len(repeat_failures) / len(failures_in_window) if failures_in_window else 0.0

    def _compute_window_success_rate(self, window: List[Dict[str, Any]]) -> float:
        """Compute success rate for a window."""
        successes = sum(1 for t in window if t.get('status') == 'success')
        return successes / len(window) if window else 0.0

    def _compute_trend(self, values: List[float]) -> str:
        """Determine if values are improving, degrading, or stable."""

        if len(values) < 3:
            return 'insufficient_data'

        # Linear regression
        x = np.arange(len(values))
        y = np.array(values)

        slope, _ = np.polyfit(x, y, 1)

        if slope < -0.05:
            return 'improving'  # RFR decreasing
        elif slope > 0.05:
            return 'degrading'  # RFR increasing
        else:
            return 'stable'

    def analyze_localization_accuracy(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure how accurately failures are localized to responsible operators.
        """

        total = 0
        correct = 0
        by_operator = defaultdict(lambda: {'total': 0, 'correct': 0})

        for trace in traces:
            if trace.get('status') != 'failure':
                continue

            actual_op = trace.get('actual_failing_operator')
            localized_op = trace.get('localized_operator')

            if actual_op and localized_op:
                total += 1
                is_correct = (actual_op == localized_op)

                if is_correct:
                    correct += 1

                by_operator[actual_op]['total'] += 1
                if is_correct:
                    by_operator[actual_op]['correct'] += 1

        accuracy = (correct / total) if total > 0 else 0

        operator_accuracy = {}
        for op, counts in by_operator.items():
            operator_accuracy[op] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0

        return {
            'overall_accuracy': accuracy,
            'total_localizations': total,
            'correct_localizations': correct,
            'by_operator': operator_accuracy,
        }

    def export_for_paper(self, rfr_breakdown: Dict[str, Any],
                        learning_curve: Dict[str, Any],
                        localization_analysis: Dict[str, Any]):
        """Export analysis results in paper-ready format."""

        # CSV: RFR breakdown
        breakdown_file = self.output_dir / 'rfr_breakdown.csv'
        with open(breakdown_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Count', 'Percentage', 'Description'])

            for category, data in rfr_breakdown['breakdown'].items():
                writer.writerow([
                    category,
                    data['count'],
                    f"{data['percentage']:.1f}",
                    data['description']
                ])

        # CSV: Learning curve
        learning_file = self.output_dir / 'learning_curve.csv'
        with open(learning_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Window', 'RFR', 'Success_Rate'])

            for i, rfr in enumerate(learning_curve['rfr_over_time']):
                sr = learning_curve['success_rate_over_time'][i]
                writer.writerow([learning_curve['windows'][i], rfr, sr])

        # CSV: Localization accuracy
        localization_file = self.output_dir / 'localization_accuracy.csv'
        with open(localization_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operator', 'Accuracy'])

            for op, acc in localization_analysis['by_operator'].items():
                writer.writerow([op, f"{acc:.2f}"])

        # JSON: Complete analysis
        summary_file = self.output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'rfr_breakdown': rfr_breakdown,
                'learning_curve': learning_curve,
                'localization_analysis': localization_analysis,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)

        print(f"ANALYSIS: Exported results to {self.output_dir}")
        print(f"  - {breakdown_file.name}")
        print(f"  - {learning_file.name}")
        print(f"  - {localization_file.name}")
        print(f"  - {summary_file.name}")

        return {
            'breakdown_csv': str(breakdown_file),
            'learning_csv': str(learning_file),
            'localization_csv': str(localization_file),
            'summary_json': str(summary_file),
        }
