import pandas as pd
from datetime import datetime
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import preprocess_data
from simulation.tariffs import get_tod_tariff_multiplier, is_solar_available
from simulation.logic import BatteryEnvironment
from agents.graph import build_smartsaver_graph

def run_simulation(month_start='2023-06-01', month_end='2023-06-30'):
    print(f"Running SmartSaver Multi-Agent Simulation ({month_start} to {month_end})...")
    
    # Needs to bring in the unscaled market data to run costs properly
    from data.scraper import get_historical_market_data
    from data.weather import get_weather_data
    market_df = get_historical_market_data(month_start, month_end)
    weather_df = get_weather_data(month_start, month_end)
    df = pd.merge(market_df, weather_df, on='timestamp', how='inner')
    
    # We will simulate the graph decisions
    battery = BatteryEnvironment(capacity_kwh=10.0, max_charge_rate_kw=5.0, initial_soc=50.0)
    graph = build_smartsaver_graph()
    
    results = []
    
    traditional_cost = 0.0
    smart_cost = 0.0
    
    print("Orchestrating agents across blocks...")
    # Simulate first 100 blocks (approx 1 day) for speed in demo, or all for reality
    # We loop over exactly what is needed
    for i in tqdm(range(len(df) - 24)): # Need 24 window for dummy forecast lookup
        row = df.iloc[i]
        
        timestamp = row['timestamp']
        grid_price = row['price_rs_per_kwh'] * get_tod_tariff_multiplier(timestamp)
        cloud_pct = row['cloud_cover_pct']
        
        solar_kw = 3.0 if is_solar_available(timestamp, cloud_pct) else 0.0
        load_kw = 2.0 # Constant average home load
        
        # Look ahead dummy "Forecast" from our dataframe
        # In actual deployment, this would be an inference call to the LSTM `model/validate.py`
        lstm_future_prices = df.iloc[i:i+24]['price_rs_per_kwh'].values.tolist()
        lstm_future_demand = df.iloc[i:i+24]['demand_mw'].values.tolist()
        
        state = {
            "timestamp": str(timestamp),
            "battery_percentage": battery.get_soc(),
            "current_grid_price": grid_price,
            "lstm_price_forecast": lstm_future_prices,
            "lstm_demand_forecast": lstm_future_demand,
            "solar_available": solar_kw > 0.0,
            "analyst_warning": "",
            "strategist_decision": "",
            "strategist_reasoning": "",
            "final_action": ""
        }
        
        # Run LangGraph Workflow
        agent_out = graph.invoke(state)
        action = agent_out['final_action']
        
        # Step Physics Engine
        grid_energy_used = battery.step(action, load_kw=load_kw, solar_kw=solar_kw)
        
        cost_step = grid_energy_used * grid_price
        
        # Traditional Home logic: always IDLE (no battery)
        trad_grid_used = max(0, load_kw * 0.25 - solar_kw * 0.25)
        trad_cost_step = trad_grid_used * grid_price
        
        traditional_cost += trad_cost_step
        smart_cost += cost_step
        
        results.append({
            'timestamp': timestamp,
            'grid_price': grid_price,
            'battery_soc': battery.get_soc(),
            'action': action,
            'reasoning': agent_out['strategist_reasoning'],
            'smart_cost': smart_cost,
            'trad_cost': traditional_cost
        })
        
    print(f"\nSimulation Complete:")
    print(f"Traditional Cost: Rs {traditional_cost:.2f}")
    print(f"SmartSaver Cost:  Rs {smart_cost:.2f}")
    
    if traditional_cost > 0:
        savings = ((traditional_cost - smart_cost) / traditional_cost) * 100
        print(f"ROI Percentage: {savings:.2f}%")
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
    res_df.to_csv(os.path.join(os.path.dirname(__file__), 'output', 'sim_results.csv'), index=False)
    print("Saved results to simulation/output/sim_results.csv")
    
if __name__ == "__main__":
    run_simulation()
