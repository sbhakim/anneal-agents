# src/governance/provenance.py
"""
Implements provenance tracking for all proposed and committed edits.
Corresponds to Section IX-A of the paper.
"""
import json
import hashlib
import time
from typing import Dict, Any, Optional
from pathlib import Path


class ProvenanceTracker:
    """
    Creates and stores immutable provenance tuples for auditability.
    """

    def __init__(self, config: Dict[str, Any]):
        self.log_path = Path(config.get('log_path', 'data/logs/provenance.jsonl'))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.source = config.get("source", "FDKA-v1.2")
        print(f"PROVENANCE: Logging to {self.log_path}")

    @staticmethod
    def _stable_hash(obj: Any) -> str:
        try:
            s = json.dumps(obj, sort_keys=True, default=str)
        except Exception:
            s = str(obj)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def create_provenance_tuple(self, patch: Dict, trace: Dict, context: Dict) -> Dict:
        """
        Generates a provenance tuple for a proposed edit Δo.
        Based on Eq. 19 from the paper (PoC-friendly).
        """
        # Prefer upstream patch ids when present for joinability across subsystems.
        patch_uuid = patch.get("id") or patch.get("patch_id")
        if not patch_uuid:
            patch_uuid = self._stable_hash(patch)[:12]

        prompt_hash = self._stable_hash({"trace": trace, "context": context})
        context_hash = self._stable_hash(context)

        prov_tuple = {
            "patch_id": patch_uuid,
            "source": self.source,
            "prompt_hash": prompt_hash,
            "context_hash": context_hash,
            "rationale": patch.get("justification", "N/A"),
            "timestamp": time.time(),
            "trace_id": trace.get("trace_id", "N/A"),
            "patch_details": patch,
            # Optional audit fields (filled when provided by caller)
            "operator": patch.get("operator"),
            "action": patch.get("action"),
            "signature": patch.get("signature") or patch.get("sig") or "",
            "edit_key": patch.get("edit_key") or "",
            "polarity": patch.get("polarity") or "",
            "decision": patch.get("decision") or "",          # applied|skipped|rejected|rollback
            "reason": patch.get("reason") or "",              # duplicate|conflict|unsat|...
            "conflict_with": patch.get("conflict_with") or "",
            "z3": patch.get("z3") or {},                      # {verdict, timeout_ms, atoms,...}
        }
        return prov_tuple

    def log(self, provenance_tuple: Dict):
        """Appends a provenance tuple to the append-only log (one JSON object per line)."""
        record = dict(provenance_tuple or {})
        if "timestamp" not in record:
            record["timestamp"] = time.time()
        if "patch_id" not in record or not record["patch_id"]:
            record["patch_id"] = self._stable_hash(record)[:12]

        line = json.dumps(record, sort_keys=True, default=str)
        with open(self.log_path, 'a', encoding="utf-8") as f:
            f.write(line + '\n')
            f.flush()
        print(f"PROVENANCE: Logged patch {record.get('patch_id')}.")

    def log_patch_event(
        self,
        patch: Dict[str, Any],
        *,
        decision: str,
        reason: str = "",
        trace: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience helper for consistent audit records in the commit path.
        """
        trace = trace or {}
        context = context or {}
        patch_record = dict(patch or {})
        patch_record["decision"] = decision
        if reason:
            patch_record["reason"] = reason
        if extra:
            patch_record.update(extra)

        prov = self.create_provenance_tuple(patch_record, trace, context)
        self.log(prov)
        return prov
