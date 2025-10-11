# src/knowledge/experience_pool.py
"""
Experience Pool (R_exp) for storing and retrieving execution traces.
Used for counterfactual replay in utility scoring (Section VIII-C).
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
        trace_id = len(self.traces)
        
        # Extract key information
        operator = metadata.get('operator', 'UNKNOWN') if metadata else 'UNKNOWN'
        error_type = metadata.get('error_type', None) if metadata else None
        
        trace_record = {
            'trace_id': trace_id,
            'trace': trace,
            'success': success,
            'operator': operator,
            'error_type': error_type,
            'metadata': metadata or {}
        }
        
        # Add to main storage
        self.traces.append(trace_record)
        
        # Update indices
        self.by_operator[operator].append(trace_id)
        if error_type:
            self.by_error_type[error_type].append(trace_id)
        self.by_success[success].append(trace_id)
        
        # Enforce max size (FIFO eviction)
        if len(self.traces) > self.max_size:
            self._evict_oldest()
        
        if trace_id % 50 == 0:
            print(f"EXPERIENCE_POOL: {len(self.traces)} traces stored")
    
    def retrieve_similar(self, failure_info: Dict[str, Any], k: int = 20) -> List[Dict]:
        """
        Retrieve k most similar traces for utility scoring.
        
        This implements the hybrid similarity metric from Section VIII-C:
        - Symbolic features (operator name, error type)
        - Predicate overlap
        
        Args:
            failure_info: Current failure information
            k: Number of traces to retrieve
            
        Returns:
            List of similar trace records
        """
        operator = failure_info.get('operator', 'UNKNOWN')
        error_type = failure_info.get('error', None)
        
        # Get candidate traces
        candidates = []
        
        # Priority 1: Same operator + same error
        if error_type:
            matching_indices = set(self.by_operator.get(operator, [])) & \
                             set(self.by_error_type.get(error_type, []))
            candidates.extend([self.traces[i] for i in matching_indices])
        
        # Priority 2: Same operator (any error)
        if len(candidates) < k:
            operator_indices = self.by_operator.get(operator, [])
            for idx in operator_indices:
                if self.traces[idx] not in candidates:
                    candidates.append(self.traces[idx])
        
        # Priority 3: Same error type (any operator)
        if len(candidates) < k and error_type:
            error_indices = self.by_error_type.get(error_type, [])
            for idx in error_indices:
                if self.traces[idx] not in candidates:
                    candidates.append(self.traces[idx])
        
        # Return top k
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
            'total_traces': len(self.traces),
            'success_count': len(self.by_success[True]),
            'failure_count': len(self.by_success[False]),
            'operators': list(self.by_operator.keys()),
            'error_types': list(self.by_error_type.keys())
        }
    
    def save(self, filepath: Path) -> None:
        """Save experience pool to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.traces, f, indent=2)
        
        print(f"EXPERIENCE_POOL: Saved {len(self.traces)} traces to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load experience pool from file."""
        if not filepath.exists():
            print(f"EXPERIENCE_POOL: No file found at {filepath}")
            return
        
        with open(filepath, 'r') as f:
            loaded_traces = json.load(f)
        
        # Rebuild pool
        self.traces = []
        self.by_operator.clear()
        self.by_error_type.clear()
        self.by_success.clear()
        
        for trace_record in loaded_traces:
            # Re-add to rebuild indices
            trace = trace_record['trace']
            success = trace_record['success']
            metadata = trace_record.get('metadata', {})
            self.add_trace(trace, success, metadata)
        
        print(f"EXPERIENCE_POOL: Loaded {len(self.traces)} traces from {filepath}")
    
    def _evict_oldest(self) -> None:
        """Evict oldest trace when max_size is exceeded."""
        if len(self.traces) > 0:
            removed = self.traces.pop(0)
            # Note: In production, would also update indices
            print(f"EXPERIENCE_POOL: Evicted oldest trace (id={removed['trace_id']})")


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
            'task_id': i,
            'operator': 'BookHotel',
            'error_type': 'POLICY_VIOLATION' if not success else None
        }
        
        pool.add_trace(trace, success, metadata)
    
    # Retrieve similar traces
    print("\n[Retrieving similar traces...]")
    failure_info = {'operator': 'BookHotel', 'error': 'POLICY_VIOLATION'}
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
