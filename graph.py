from langgraph.graph import StateGraph, START, END
from .state import SmartSaverState
from .nodes import analyst_node, strategist_node, safety_guard_node

def build_smartsaver_graph():
    """
    Constructs the LangGraph workflow:
    Analyst -> Strategist -> Safety Guard
    """
    workflow = StateGraph(SmartSaverState)
    
    # Add nodes
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("safety_guard", safety_guard_node)
    
    # Add edges
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", "strategist")
    workflow.add_edge("strategist", "safety_guard")
    workflow.add_edge("safety_guard", END)
    
    # Compile
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Test the graph with dummy state
    graph = build_smartsaver_graph()
    
    initial_state = {
        "timestamp": "2024-02-15 14:00:00",
        "battery_percentage": 15.0,
        "current_grid_price": 5.0,
        "lstm_price_forecast": [5.5, 6.0, 10.5, 12.0], # Peak coming!
        "lstm_demand_forecast": [150000, 155000, 160000, 165000],
        "solar_available": False,
        "analyst_warning": "",
        "strategist_decision": "",
        "strategist_reasoning": "",
        "final_action": ""
    }
    
    print("Running Graph on initial state...")
    result = graph.invoke(initial_state)
    
    print("\n--- Final Agent State ---")
    print(f"Analyst: {result['analyst_warning']}")
    print(f"Strategist Intent: {result['strategist_decision']}")
    print(f"Strategist Reasoning: {result['strategist_reasoning']}")
    print(f"Final Action: {result['final_action']}")
