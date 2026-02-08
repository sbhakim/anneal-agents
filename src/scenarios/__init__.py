"""Scenario implementations for different evaluation domains."""

from .travel_planning import TravelPlanningScenario
from .ecommerce import EcommerceScenario
from .travel_planning_stochastic import StochasticTravelScenario

SCENARIO_MAP = {
    'travel': TravelPlanningScenario,
    'travel_planning': TravelPlanningScenario,
    'ecommerce': EcommerceScenario,
    'travel_stochastic': StochasticTravelScenario,
}

__all__ = [
    'TravelPlanningScenario',
    'EcommerceScenario',
    'StochasticTravelScenario',
    'SCENARIO_MAP',
]
