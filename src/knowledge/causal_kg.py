# src/knowledge/causal_kg.py
"""
Implements the Causal Knowledge Graph (KG_cau) for structural relationships and hazards.
Models cause-effect links, intervention effects, and detects ambiguities (Section II-C).

MINIMUM RELIABILITY UPDATE (2026-01):
- Fix overly-broad blackout/payment hazard that was triggering on *any* corporate card
  whenever 'blackout_dates' existed in state. That caused governance gridlock.
- Hazard now fires ONLY when:
  (a) blackout policy is active AND
  (b) travel_dates overlaps blackout_dates AND
  (c) payment is invalid (explicit flag or '(invalid)' marker or payment_valid False)
This keeps the system safe without blocking unrelated retry patches (e.g., ApiTimeoutRetry).
"""
from typing import Dict, Any, List, Tuple, Callable
import networkx as nx  # For graph-based causal modeling (pre-installed in environment)


class CausalKG:
    """
    A directed graph encoding causal relationships, hazards, and intervention effects.
    Supports adding edges (causes) and querying for hazards or effects during verification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.graph = nx.DiGraph()  # Directed graph for causal flows
        self.hazards: Dict[str, Callable[[Dict[str, Any]], bool]] = {}  # hazard_id -> detection fn
        self.interventions: Dict[str, List[Tuple[str, str]]] = {}  # action -> [(cause, effect)]
        self._load_default_causals()
        print(
            f"CAUSAL_KG: Initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

    # ---------------------------------------------------------------------
    # Helpers (minimal, local)
    # ---------------------------------------------------------------------

    def _is_payment_invalid(self, state: Dict[str, Any]) -> bool:
        """
        Determines if current payment should be treated as invalid.
        Uses both explicit flags and string markers for compatibility.
        """
        if state.get("payment_invalid") is True:
            return True
        if state.get("payment_valid") is False:
            return True
        pm = state.get("payment_method", "")
        if isinstance(pm, str) and "invalid" in pm.lower():
            return True
        return False

    def _normalize_date_token(self, s: str) -> str:
        """Normalize date tokens for rough matching (e.g., 'Apr 10', 'April 10')."""
        if not isinstance(s, str):
            return ""
        return " ".join(s.strip().split()).lower()

    def _extract_dates_from_travel_dates(self, travel_dates: Any) -> List[str]:
        """
        Extracts a list of date tokens from the state's travel_dates string.
        Supported formats from this codebase:
          - "Apr 10-12" (range)
          - "Apr 10" (single)
        Returns list of normalized tokens like ["apr 10", "apr 11", ...] or ["apr 10"].
        """
        if not isinstance(travel_dates, str) or not travel_dates.strip():
            return []

        td = travel_dates.strip()
        td_norm = self._normalize_date_token(td)

        # Range like "Apr 10-12"
        if "-" in td_norm:
            # naive parse: "<mon> <start>-<end>"
            # Example: "apr 10-12"
            try:
                left, right = td_norm.split("-", 1)
                left = left.strip()  # "apr 10"
                right = right.strip()  # "12"
                parts = left.split()
                if len(parts) >= 2 and parts[-1].isdigit() and right.isdigit():
                    mon = " ".join(parts[:-1])
                    start = int(parts[-1])
                    end = int(right)
                    if end < start:
                        start, end = end, start
                    return [self._normalize_date_token(f"{mon} {d}") for d in range(start, end + 1)]
            except Exception:
                # fallback to best-effort token match
                return [td_norm]

        # Single date like "on Apr 10" should already be stored as "Apr 10" by parser
        return [td_norm]

    def _dates_overlap(self, travel_dates: Any, blackout_dates: Any) -> bool:
        """
        Returns True if travel_dates overlaps blackout_dates.
        blackout_dates is expected to be a list like ["Apr 10", "Apr 11", ...].
        """
        travel_tokens = self._extract_dates_from_travel_dates(travel_dates)
        if not travel_tokens:
            return False

        if not isinstance(blackout_dates, list) or not blackout_dates:
            return False

        blackout_tokens = {self._normalize_date_token(x) for x in blackout_dates if isinstance(x, str)}
        return any(t in blackout_tokens for t in travel_tokens)

    # ---------------------------------------------------------------------
    # Default graph + hazards
    # ---------------------------------------------------------------------

    def _load_default_causals(self) -> None:
        """
        Loads default causal relations for the travel domain (e.g., payment -> booking success).
        """
        # Nodes: Key entities/actions in domain
        nodes = ['payment_method', 'blackout_dates', 'booking_attempt', 'api_health', 'success', 'timeout_error']
        self.graph.add_nodes_from(nodes)

        # Edges: Causal links (e.g., invalid payment causes failure)
        self.graph.add_edge('payment_method', 'success', weight=0.8)  # Positive causal strength
        self.graph.add_edge('blackout_dates', 'success', weight=-0.6)  # Negative (hazard)
        self.graph.add_edge('api_health', 'timeout_error', weight=-0.7)
        self.graph.add_edge('timeout_error', 'success', weight=-0.9)

        # Hazard: Detect invalid payment during blackout (policy-relevant)
        # FIX: previously triggered whenever blackout_dates key existed + corporate payment, causing false positives.
        def hazard_blackout_invalid(state: Dict[str, Any]) -> bool:
            blackout_dates = state.get("blackout_dates", [])
            travel_dates = state.get("travel_dates")
            policy = state.get("corporate_card_policy", "")
            policy_active = (policy == "blocked_on_blackout_dates")

            if not policy_active:
                return False

            # Only relevant if the trip overlaps a blackout date
            if not self._dates_overlap(travel_dates, blackout_dates):
                return False

            # Only a hazard if payment is invalid in this context
            if not self._is_payment_invalid(state):
                return False

            return True

        self.hazards['blackout_payment_hazard'] = hazard_blackout_invalid

        # Intervention: e.g., RetryOnTimeout affects timeout_error -> success
        self.interventions['RetryOnTimeout'] = [('timeout_error', 'success')]
        # Keep a compatible alias for patches that use ApiTimeoutRetry naming
        self.interventions['ApiTimeoutRetry'] = [('timeout_error', 'success')]

    def add_relation(self, cause: str, effect: str, weight: float = 1.0) -> None:
        """
        Adds a causal edge (e.g., from FDKA patches or policy updates).
        """
        self.graph.add_edge(cause, effect, weight=weight)
        print(f"CAUSAL_KG: Added relation {cause} -> {effect} (weight={weight})")

    def add_hazard(self, hazard_id: str, detection_fn: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Adds a hazard detection function.
        """
        if hazard_id in self.hazards:
            print(f"CAUSAL_KG: Hazard '{hazard_id}' already exists. Overwriting.")
        self.hazards[hazard_id] = detection_fn
        print(f"CAUSAL_KG: Added hazard '{hazard_id}'")

    def check_hazards(self, state: Dict[str, Any]) -> bool:
        """
        Checks all hazards for potential issues.
        Returns True if no hazards detected, False otherwise.
        Used in verification/guardrails (Section VIII-E).
        """
        for hazard_id, detect in self.hazards.items():
            try:
                if detect(state):
                    print(f"CAUSAL_KG: Hazard detected in '{hazard_id}'")
                    return False
            except Exception as e:
                print(f"CAUSAL_KG: Error in hazard '{hazard_id}': {e}")
                return False  # Fail-safe on error
        return True

    def get_intervention_effects(self, action: str, state: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Returns potential effects of an intervention (e.g., for counterfactual scoring).
        """
        if action in self.interventions:
            effects = self.interventions[action]
            # Filter based on state (mock: return all for PoC)
            return effects
        return []

    def detect_cycles(self) -> bool:
        """
        Checks for causal cycles (ambiguities) using networkx.
        """
        try:
            cycles = list(nx.find_cycle(self.graph))
            if cycles:
                print(f"CAUSAL_KG: Cycles detected: {cycles}")
                return True
            return False
        except nx.NetworkXNoCycle:
            return False
