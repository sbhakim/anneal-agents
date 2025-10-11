# src/governance/provenance.py
"""
Implements provenance tracking for all proposed and committed edits.
Corresponds to Section IX-A of the paper.
"""
import json
import hashlib
import time
from typing import Dict, Any, List
from pathlib import Path


class ProvenanceTracker:
    """
    Creates and stores immutable provenance tuples for auditability.
    """

    def __init__(self, config: Dict[str, Any]):
        self.log_path = Path(config.get('log_path', 'data/logs/provenance.jsonl'))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"PROVENANCE: Logging to {self.log_path}")

    def create_provenance_tuple(self, patch: Dict, trace: Dict, context: Dict) -> Dict:
        """
        Generates a provenance tuple for a proposed edit Δo.
        Based on Eq. 19 from the paper.
        """
        prompt_str = f"{trace}{context}"

        prov_tuple = {
            "patch_id": hashlib.sha256(str(patch).encode()).hexdigest()[:12],
            "source": "FDKA-v1.2",
            "prompt_hash": hashlib.sha256(prompt_str.encode()).hexdigest(),
            "context_hash": hashlib.sha256(str(context).encode()).hexdigest(),
            "rationale": patch.get("justification", "N/A"),
            "timestamp": time.time(),
            "trace_id": trace.get("trace_id", "N/A"),
            "patch_details": patch
        }
        return prov_tuple

    def log(self, provenance_tuple: Dict):
        """Appends a provenance tuple to the append-only log."""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(provenance_tuple) + '\n')
        print(f"PROVENANCE: Logged patch {provenance_tuple['patch_id']}.")