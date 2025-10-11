# src/knowledge/causal_kg.py
"""
Implements the Causal Knowledge Graph (KG_cau) for structural relationships and hazards.
Models cause-effect links, intervention effects, and detects ambiguities (Section II-C).
"""
from typing import Dict, Any, List, Tuple, Callable
import networkx as nx  # For graph-based causal modeling (pre-installed in environment)


class CausalKG:
    """
    A directed graph encoding causal relationships, hazards, and intervention effects.
    Supports adding edges (causes) and querying for hazards or effects during verification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.graph = nx.DiGraph()  # Directed graph for causal flows
        self.hazards: Dict[str, Callable[[Dict[str, Any]], bool]] = {}  # hazard_id -> detection fn
        self.interventions: Dict[str, List[Tuple[str, str]]] = {}  # action -> [(cause, effect)]
        self._load_default_causals()
        print(f"CAUSAL_KG: Initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

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

        # Hazard: Detect invalid payment during blackout
        def hazard_blackout_invalid(state: Dict[str, Any]) -> bool:
            return 'blackout_dates' in state and state.get('payment_method', '').startswith('Corporate')

        self.hazards['blackout_payment_hazard'] = hazard_blackout_invalid

        # Intervention: e.g., RetryOnTimeout affects timeout_error -> success
        self.interventions['RetryOnTimeout'] = [('timeout_error', 'success')]

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