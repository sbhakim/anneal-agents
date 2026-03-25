"""Scenario implementations for different evaluation domains."""

from .travel_planning import TravelPlanningScenario
from .ecommerce import EcommerceScenario
from .travel_planning_stochastic import StochasticTravelScenario
from .itsm import ITSMScenario

SCENARIO_MAP = {
    'travel': TravelPlanningScenario,
    'travel_planning': TravelPlanningScenario,
    'ecommerce': EcommerceScenario,
    'travel_stochastic': StochasticTravelScenario,
    'itsm': ITSMScenario,
}

__all__ = [
    'TravelPlanningScenario',
    'EcommerceScenario',
    'StochasticTravelScenario',
    'ITSMScenario',
    'SCENARIO_MAP',
]
