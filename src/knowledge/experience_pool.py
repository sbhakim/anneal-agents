# src/knowledge/experience_pool.py
"""
Experience Pool (R_exp) for storing and retrieving execution traces.
Used for counterfactual replay in utility scoring (Section VIII-C).

MINIMAL RELIABILITY UPDATES:
- Infer operator/error_type from the trace when metadata is missing (common for success traces).
  This makes get_success_traces(operator=...) actually work for canary balancing.
- Fix FIFO eviction to keep indices consistent (reindex + rebuild).
- Load now rebuilds indices directly (avoids double-indexing / drift).
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
from pathlib import Path


class ExperiencePool:
    """
    Stores execution traces for retrieval-based patch evaluation.
    Corresponds to R_exp in the paper's formalization.
    """

    def __init__(self, max_size: int = 1000):
        self.traces: List[Dict[str, Any]] = []
        self.max_size = max_size

        # Index by operator and error type for fast retrieval
        # NOTE: indices store list positions (trace_record list index), so eviction must reindex.
        self.by_operator: Dict[str, List[int]] = defaultdict(list)
        self.by_error_type: Dict[str, List[int]] = defaultdict(list)
        self.by_success: Dict[bool, List[int]] = defaultdict(list)

        print(f"EXPERIENCE_POOL: Initialized (max_size={max_size})")

    def add_trace(self, trace: List[Dict], success: bool, metadata: Optional[Dict] = None) -> None:
        """
        Add an execution trace to the pool.

        Args:
            trace: Execution trace (list of steps)
            success: Whether execution succeeded
            metadata: Additional metadata (task_id, operator, etc.)
        """
        metadata = metadata or {}

        # Operator/error_type are required for meaningful retrieval; infer when missing.
        operator = metadata.get("operator") or self._infer_operator_from_trace(trace) or "UNKNOWN"
        error_type = metadata.get("error_type")
        if error_type is None:
            error_type = self._infer_error_type_from_trace(trace)

        trace_record = {
            "trace_id": len(self.traces),
            "trace": trace,
            "success": success,
            "operator": operator,
            "error_type": error_type,
            "metadata": metadata,
        }

        self.traces.append(trace_record)
        self._index_trace(len(self.traces) - 1, trace_record)

        # Enforce max size (FIFO eviction)
        if len(self.traces) > self.max_size:
            self._evict_oldest()

        if len(self.traces) % 50 == 0:
            print(f"EXPERIENCE_POOL: {len(self.traces)} traces stored")

    def retrieve_similar(self, failure_info: Dict[str, Any], k: int = 20) -> List[Dict]:
        """
        Retrieve k most similar traces for utility scoring.

        This implements a hybrid similarity metric:
        - Symbolic features (operator name, error type)

        Args:
            failure_info: Current failure information
            k: Number of traces to retrieve

        Returns:
            List of similar trace records
        """
        operator = failure_info.get("operator", "UNKNOWN")
        error_type = failure_info.get("error", None)

        candidates: List[Dict[str, Any]] = []

        # Priority 1: Same operator + same error
        if error_type:
            matching_indices = set(self.by_operator.get(operator, [])) & set(self.by_error_type.get(error_type, []))
            candidates.extend([self.traces[i] for i in matching_indices])

        # Priority 2: Same operator (any error)
        if len(candidates) < k:
            operator_indices = self.by_operator.get(operator, [])
            for idx in operator_indices:
                tr = self.traces[idx]
                if tr not in candidates:
                    candidates.append(tr)

        # Priority 3: Same error type (any operator)
        if len(candidates) < k and error_type:
            error_indices = self.by_error_type.get(error_type, [])
            for idx in error_indices:
                tr = self.traces[idx]
                if tr not in candidates:
                    candidates.append(tr)

        result = candidates[:k]
        print(f"EXPERIENCE_POOL: Retrieved {len(result)} similar traces for {operator}:{error_type}")
        return result

    def get_failure_traces(self, operator: Optional[str] = None) -> List[Dict]:
        """
        Get all failure traces, optionally filtered by operator.

        Args:
            operator: Optional operator name filter

        Returns:
            List of failure trace records
        """
        failure_indices = self.by_success[False]

        if operator:
            operator_indices = set(self.by_operator.get(operator, []))
            failure_indices = [i for i in failure_indices if i in operator_indices]

        return [self.traces[i] for i in failure_indices]

    def get_success_traces(self, operator: Optional[str] = None) -> List[Dict]:
        """Get all successful traces, optionally filtered by operator."""
        success_indices = self.by_success[True]

        if operator:
            operator_indices = set(self.by_operator.get(operator, []))
            success_indices = [i for i in success_indices if i in operator_indices]

        return [self.traces[i] for i in success_indices]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the experience pool."""
        return {
            "total_traces": len(self.traces),
            "success_count": len(self.by_success[True]),
            "failure_count": len(self.by_success[False]),
            "operators": list(self.by_operator.keys()),
            "error_types": list(self.by_error_type.keys()),
        }

    def save(self, filepath: Path) -> None:
        """Save experience pool to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.traces, f, indent=2)

        print(f"EXPERIENCE_POOL: Saved {len(self.traces)} traces to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load experience pool from file."""
        if not filepath.exists():
            print(f"EXPERIENCE_POOL: No file found at {filepath}")
            return

        with open(filepath, "r") as f:
            loaded_traces = json.load(f)

        # Keep most recent traces if the file exceeds max_size.
        if isinstance(loaded_traces, list) and len(loaded_traces) > self.max_size:
            loaded_traces = loaded_traces[-self.max_size :]

        self.traces = loaded_traces if isinstance(loaded_traces, list) else []
        self._reindex_and_rebuild()

        print(f"EXPERIENCE_POOL: Loaded {len(self.traces)} traces from {filepath}")

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _index_trace(self, idx: int, trace_record: Dict[str, Any]) -> None:
        operator = trace_record.get("operator", "UNKNOWN")
        error_type = trace_record.get("error_type", None)
        success = bool(trace_record.get("success", False))

        self.by_operator[operator].append(idx)
        if error_type:
            self.by_error_type[error_type].append(idx)
        self.by_success[success].append(idx)

    def _reindex_and_rebuild(self) -> None:
        """Indices store list positions; keep them consistent after load/eviction."""
        self.by_operator.clear()
        self.by_error_type.clear()
        self.by_success.clear()

        for i, tr in enumerate(self.traces):
            # Normalize trace_id to current list index (internal identity)
            if isinstance(tr, dict):
                tr["trace_id"] = i
            self._index_trace(i, tr)

    def _infer_operator_from_trace(self, trace: List[Dict]) -> Optional[str]:
        # Prefer an explicit operator field in any trace entry; use the last one seen.
        for entry in reversed(trace or []):
            if not isinstance(entry, dict):
                continue
            op = entry.get("operator")
            if op:
                return getattr(op, "name", str(op))
            # Some traces use "step" as operator-ish label.
            step = entry.get("step")
            if step and isinstance(step, str):
                return step
        return None

    def _infer_error_type_from_trace(self, trace: List[Dict]) -> Optional[str]:
        # Failures generally include an 'error' field; use the last error in the trace.
        for entry in reversed(trace or []):
            if not isinstance(entry, dict):
                continue
            if "error" in entry:
                return entry.get("error")
            if "error_type" in entry:
                return entry.get("error_type")
        return None

    def _evict_oldest(self) -> None:
        """Evict oldest trace when max_size is exceeded; keep indices consistent."""
        if not self.traces:
            return

        removed = self.traces.pop(0)
        self._reindex_and_rebuild()
        rid = removed.get("trace_id", "?") if isinstance(removed, dict) else "?"
        print(f"EXPERIENCE_POOL: Evicted oldest trace (id={rid})")


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ExperiencePool")
    print("=" * 60)

    pool = ExperiencePool(max_size=10)

    # Add some traces
    print("\n[Adding traces...]")
    for i in range(5):
        trace = [{"step": "BookHotel", "state": {}}]
        success = i % 2 == 0

        if not success:
            trace.append({"error": "POLICY_VIOLATION", "operator": "BookHotel"})

        metadata = {
            "task_id": i,
            "operator": "BookHotel",
            "error_type": "POLICY_VIOLATION" if not success else None,
        }

        pool.add_trace(trace, success, metadata)

    # Retrieve similar traces
    print("\n[Retrieving similar traces...]")
    failure_info = {"operator": "BookHotel", "error": "POLICY_VIOLATION"}
    similar = pool.retrieve_similar(failure_info, k=3)
    print(f"Found {len(similar)} similar traces")

    # Get stats
    print("\n[Pool statistics]")
    stats = pool.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save and load
    print("\n[Testing save/load...]")
    test_path = Path("test_experience_pool.json")
    pool.save(test_path)

    new_pool = ExperiencePool()
    new_pool.load(test_path)
    print(f"Loaded pool has {len(new_pool.traces)} traces")

    print("\n✅ ExperiencePool test complete")
