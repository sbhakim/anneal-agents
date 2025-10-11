# src/core/parser.py
"""
Implements the neural semantic parser for grounding symbolic plans.
Maps natural language or symbolic parameters to concrete entities/literals (Section II-A).
"""
from typing import Dict, Any, List
import re  # For regex-based grounding (mock neural parser)
import numpy as np  # <- added so compute_uncertainty works


class Parser:
    """
    A neural semantic parser that grounds operator parameters to the current context.
    For PoC, uses regex mocks; in full impl, integrate LLM with constrained prompts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('propose_edit', {}).get('model', 'gpt-4')  # Placeholder for LLM model
        self.temperature = config.get('propose_edit', {}).get('temperature', 0.3)
        print(f"PARSER: Initialized with model '{self.model}' and temperature {self.temperature}")

    def ground_params(self, instruction: str, operator_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grounds parameters for a specific operator based on the instruction and state.
        Returns a dict of param_name -> grounded_value.
        """
        params = {}

        if operator_name == "BookHotel":
            params = self._ground_hotel_params(instruction, state)
        elif operator_name == "BookFlight":
            params = self._ground_flight_params(instruction, state)
        else:
            print(f"PARSER: Unknown operator '{operator_name}'. No grounding applied.")

        if params:
            print(f"PARSER: Grounded params for '{operator_name}': {params}")
        else:
            print(f"PARSER: Failed to ground params for '{operator_name}'.")

        return params

    def _ground_hotel_params(self, instruction: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock neural grounding for BookHotel params using regex (PoC).
        In full impl: Use LLM prompt like "Extract location, dates, payment from: {instruction}".
        """
        params = {}

        # Location: e.g., "in San Francisco"
        loc_match = re.search(r'in\s([\w\s]+?)(?=\sfor|\son)', instruction, re.IGNORECASE)
        if loc_match:
            params['location'] = loc_match.group(1).strip()

        # Dates: e.g., "for April 10-12"
        date_match = re.search(r'(\w+\s\d+-\d+)', instruction)
        if date_match:
            params['dates'] = date_match.group(1)

        # Payment: From state or default (mock ontology grounding)
        params['payment'] = state.get("payment_method", "CorporateCard:CC-5512")

        # Full impl: LLM call, e.g., openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        # Parse JSON response for params

        return params if 'location' in params and 'dates' in params else {}

    def _ground_flight_params(self, instruction: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock neural grounding for BookFlight params.
        """
        params = {}

        # Origin: e.g., "from Washington DC"
        origin_match = re.search(r'from\s([\w\s]+?)(?=\sto|\son)', instruction, re.IGNORECASE)
        if origin_match:
            params['origin'] = origin_match.group(1).strip()

        # Destination: e.g., "to New York"
        dest_match = re.search(r'to\s([\w\s]+?)(?=\sfor|\son)', instruction, re.IGNORECASE)
        if dest_match:
            params['destination'] = dest_match.group(1).strip()

        # Date: From state or instruction
        date_match = re.search(r'on\s(\w+\s\d+)', instruction, re.IGNORECASE)
        if date_match:
            params['date'] = date_match.group(1)
        else:
            params['date'] = state.get("travel_dates", "").split('-')[0]  # First date if range

        return params if 'origin' in params and 'destination' in params else {}

    def compute_uncertainty(self, grounded_params: Dict[str, Any]) -> float:
        """
        Computes entropy-based uncertainty for grounded params (mock for integration with signals).
        """
        # Mock: Low uncertainty if all params grounded, high otherwise
        expected_params = ['location', 'dates', 'payment']  # Example for BookHotel
        grounded_count = len(grounded_params)
        entropy = - (grounded_count / len(expected_params)) * np.log(grounded_count / len(expected_params) + 1e-10)
        return min(entropy, 1.0)  # Normalize to [0,1]
