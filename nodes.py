from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from .state import SmartSaverState
import os
from dotenv import load_dotenv
import statistics

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Node 1: The Analyst (Data Interpreter)
def analyst_node(state: SmartSaverState) -> SmartSaverState:
    """
    Analyzes the LSTM forecast for the next 24 (or 48) steps.
    Identifies if a peak price is coming soon and generates a summary string.
    """
    forecast = state.get('lstm_price_forecast', [])
    current_price = state.get('current_grid_price', 0)
    
    if not forecast:
        state['analyst_warning'] = "No forecast available."
        return state
        
    avg_future_price = statistics.mean(forecast)
    max_future_price = max(forecast)
    max_index = forecast.index(max_future_price)
    
    # Calculate roughly how many hours ahead the peak is
    hours_ahead = max_index * 0.25 # assuming 15-min intervals
    
    if max_future_price > current_price * 1.5:
        warning = f"Alert: Peak price of Rs {max_future_price:.2f}/kWh expected in ~{hours_ahead:.1f} hours. Current off-peak (Rs {current_price:.2f}/kWh)."
    elif current_price > avg_future_price * 1.2:
        warning = "Alert: We are currently in a price peak! Future prices are expected to drop."
    else:
        warning = f"Prices are stable. Average forecasted price: Rs {avg_future_price:.2f}/kWh."
        
    state['analyst_warning'] = warning
    return state

# Node 2: The Strategist (LLM Decision Maker)
def strategist_node(state: SmartSaverState) -> SmartSaverState:
    """
    Takes the Analyst's warning, current conditions, and decides on the best strategy via an LLM.
    """
    # Check if a valid API key exists
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Fallback dummy strategist if no key
        return fallback_strategist(state)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    
    prompt = PromptTemplate.from_template(
        """You are the 'Strategist' for SmartSaver, an AI-driven Virtual Power Plant.
        Your goal is to optimize battery usage to save money.
        
        Current State:
        - Time: {timestamp}
        - Battery: {battery_percentage}% (Max: 100%, Min allowed: 20%)
        - Current Grid Price: Rs {current_grid_price}/kWh
        - Solar Power Available: {solar_available}
        - Analyst Warning: {analyst_warning}
        
        Rules:
        1. If Solar is available and battery is < 95%, we should CHARGE_SOLAR.
        2. If we are in or very close to a price peak and battery > 20%, we should DISCHARGE (use battery to power home, disconnect from grid).
        3. If a price peak is coming soon and battery is low (< 50%), and solar is NOT available, we should CHARGE_GRID to prepare for the peak.
        4. Otherwise, IDLE.
        
        Provide your reasoning in one short sentence, then a NEWLINE, then your final decision.
        The decision MUST BE EXACTLY one of: CHARGE_GRID, CHARGE_SOLAR, DISCHARGE, IDLE.
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "timestamp": state['timestamp'],
        "battery_percentage": state['battery_percentage'],
        "current_grid_price": state['current_grid_price'],
        "solar_available": state['solar_available'],
        "analyst_warning": state['analyst_warning']
    })
    
    lines = response.content.strip().split('\n')
    decision = lines[-1].strip().upper()
    reasoning = " ".join(lines[:-1]).strip()
    
    # Clean up standard model weirdness
    valid_decisions = ["CHARGE_GRID", "CHARGE_SOLAR", "DISCHARGE", "IDLE"]
    if decision not in valid_decisions:
        decision = "IDLE" # fallback
        reasoning += " (Fallback due to parse error)"
        
    state['strategist_decision'] = decision
    state['strategist_reasoning'] = reasoning
    return state

def fallback_strategist(state: SmartSaverState) -> SmartSaverState:
    """Hard-coded fallback if Gemini API is not configured."""
    batt = state['battery_percentage']
    warning = state['analyst_warning']
    solar = state['solar_available']
    
    if solar and batt < 95:
        state['strategist_decision'] = "CHARGE_SOLAR"
        state['strategist_reasoning'] = "Free solar is available."
    elif "currently in a price peak" in warning and batt > 20:
        state['strategist_decision'] = "DISCHARGE"
        state['strategist_reasoning'] = "Current prices are too high, using battery."
    elif "Peak price" in warning and batt < 50:
        state['strategist_decision'] = "CHARGE_GRID"
        state['strategist_reasoning'] = "Pre-charging before upcoming peak."
    else:
        state['strategist_decision'] = "IDLE"
        state['strategist_reasoning'] = "Conditions are stable, saving battery cycles."
        
    return state

# Node 3: The Safety Guard (Hard-coded safety checker)
def safety_guard_node(state: SmartSaverState) -> SmartSaverState:
    """
    Double checks the strategist's decision.
    Prevents deep discharge below 20%.
    Prevents overcharging above 100%.
    """
    decision = state.get('strategist_decision', 'IDLE')
    battery = state.get('battery_percentage', 0)
    
    final_action = decision
    
    if decision == "DISCHARGE" and battery <= 20:
        final_action = "IDLE"
        state['strategist_reasoning'] += " [SAFETY OVERRIDE: Prevented deep discharge below 20%]"
        
    if decision in ["CHARGE_GRID", "CHARGE_SOLAR"] and battery >= 100:
        final_action = "IDLE"
        state['strategist_reasoning'] += " [SAFETY OVERRIDE: Battery already full]"
        
    state['final_action'] = final_action
    return state
