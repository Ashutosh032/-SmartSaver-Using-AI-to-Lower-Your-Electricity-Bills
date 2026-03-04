from typing import TypedDict, List, Dict, Any, Optional

class SmartSaverState(TypedDict):
    # Inputs injected into the state for each time step
    timestamp: str
    battery_percentage: float
    current_grid_price: float
    lstm_price_forecast: List[float] # Next 24 hours (or next few steps)
    lstm_demand_forecast: List[float]
    solar_available: bool
    
    # Internal agent reasoning and outputs
    analyst_warning: str
    strategist_decision: str  # "CHARGE_GRID", "CHARGE_SOLAR", "DISCHARGE", "IDLE"
    strategist_reasoning: str
    final_action: str         # After safety guard checks
    
