# src/knowledge/process_kg.py
"""
Implements the Process Knowledge Graph (PKG) for operators, constraints, and ontologies.
Used by the planner for HTN compilation and plan decomposition (Section II-A).
"""
from typing import Dict, Any, List, Tuple, Callable
import networkx as nx  # For hierarchical task network (HTN) representation


class ProcessKG:
    """
    A graph encoding operators (actions with pre/eff), constraints, and domain ontologies.
    Supports HTN-style decomposition: high-level tasks -> sub-tasks/operators.
    """

    def __init__(self, config: Dict[str, Any]):
        self.graph = nx.DiGraph()  # Directed graph for task decompositions (parent -> children)
        self.operators: Dict[str, Dict[str, Any]] = {}  # op_name -> {'params': list, 'pre': list, 'eff': list}
        self.constraints: Dict[str, Callable[[Dict[str, Any]], bool]] = {}  # constraint_id -> check fn
        self.ontologies: Dict[str, List[str]] = {}  # type -> [subtypes or instances]
        self._load_default_processes()
        print(f"PROCESS_KG: Initialized with {len(self.operators)} operators and {self.graph.number_of_edges()} decompositions")

    def _load_default_processes(self) -> None:
        """
        Loads default processes for the travel domain (e.g., TravelPlan -> BookFlight + BookHotel).
        """
        # Ontologies: Domain types
        self.ontologies['PaymentMethod'] = ['CorporateCard', 'PersonalCard']
        self.ontologies['Location'] = ['San Francisco', 'New York', 'Boston', 'Seattle', 'Washington DC', 'Philadelphia', 'Newark']

        # Operators: Basic actions (aligned with rule_pool)
        self.operators['BookHotel'] = {
            'params': ['location', 'dates', 'payment'],
            'pre': ['is_hotel_available', 'is_card_valid'],
            'eff': ['set_hotel_booked']
        }
        self.operators['BookFlight'] = {
            'params': ['origin', 'destination', 'date'],
            'pre': ['is_flight_available', 'is_card_valid'],
            'eff': ['set_flight_booked']
        }

        # Constraints: Domain invariants
        def valid_dates(state: Dict[str, Any]) -> bool:
            dates = state.get('travel_dates', '')
            return bool(dates)  # Simple check: dates must be set

        self.constraints['valid_travel_dates'] = valid_dates

        # HTN Decompositions: High-level tasks
        self.graph.add_node('TravelPlan', type='task')
        self.graph.add_node('BookHotel', type='operator')
        self.graph.add_node('BookFlight', type='operator')
        self.graph.add_edge('TravelPlan', 'BookFlight', order=1)
        self.graph.add_edge('TravelPlan', 'BookHotel', order=2)

    def add_operator(self, name: str, params: List[str], pre: List[str], eff: List[str]) -> None:
        """
        Adds a new operator to the PKG.
        """
        if name in self.operators:
            print(f"PROCESS_KG: Operator '{name}' already exists. Overwriting.")
        self.operators[name] = {'params': params, 'pre': pre, 'eff': eff}
        self.graph.add_node(name, type='operator')
        print(f"PROCESS_KG: Added operator '{name}'")

    def add_decomposition(self, task: str, sub_tasks: List[Tuple[str, int]]) -> None:
        """
        Adds HTN decomposition: task -> ordered sub-tasks/operators.
        """
        if not self.graph.has_node(task):
            self.graph.add_node(task, type='task')
        for sub, order in sub_tasks:
            self.graph.add_edge(task, sub, order=order)
        print(f"PROCESS_KG: Added decomposition for '{task}' -> {sub_tasks}")

    def add_constraint(self, constraint_id: str, check_fn: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Adds a new constraint (e.g., from FDKA or policy updates).
        """
        if constraint_id in self.constraints:
            print(f"PROCESS_KG: Constraint '{constraint_id}' already exists. Overwriting.")
        self.constraints[constraint_id] = check_fn
        print(f"PROCESS_KG: Added constraint '{constraint_id}'")

    def get_decomposition(self, task: str) -> List[str]:
        """
        Returns ordered sub-tasks/operators for a high-level task (for planner).
        """
        if not self.graph.has_node(task):
            return []
        subs = sorted(self.graph.successors(task), key=lambda n: self.graph[task][n]['order'])
        return subs

    def check_constraints(self, state: Dict[str, Any]) -> bool:
        """
        Checks all constraints for the current state.
        Returns True if all pass, False otherwise.
        Integrated with verification (Section II-A).
        """
        for cid, check in self.constraints.items():
            try:
                if not check(state):
                    print(f"PROCESS_KG: Constraint violation in '{cid}'")
                    return False
            except Exception as e:
                print(f"PROCESS_KG: Error in constraint '{cid}': {e}")
                return False  # Fail-safe
        return True

    def get_ontology(self, type_name: str) -> List[str]:
        """
        Returns subtypes or instances for a given type (for grounding).
        """
        return self.ontologies.get(type_name, [])