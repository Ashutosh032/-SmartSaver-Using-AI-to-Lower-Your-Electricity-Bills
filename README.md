 SmartSaver: ML-Infused Multi-Agent VPP

SmartSaver is an intelligent energy management system designed to transition passive electricity consumers into "Prosumers" within a **Virtual Power Plant (VPP)** framework. By leveraging high-precision price forecasting and autonomous agentic decision-making, the system optimizes Battery Energy Storage Systems (BESS) to minimize costs and maximize self-consumption

# Key Features


High-Precision Forecasting: Dual-stage pipeline comparing **XGBoost** and **Hybrid CNN-LSTM** architectures to predict 96 daily IEX price blocks.


 
Agentic Decision Layer: Uses **LangGraph** to autonomously trigger battery charge/discharge cycles based on market forecasts.



Robust Data Engineering: Automated sanitization of chronological gaps and Min-Max scaling to handle localized market volatility.


  
Real-Time Edge Deployment: Optimized for minimal inference latency (<15ms), making it viable for low-power smart home gateways.






The project has successfully established a baseline for price forecasting using **Indian Energy Exchange (IEX)** data.



#System Architecture

The framework operates as a horizontal multi-stage pipeline:

Raw Ingestion: Automated web scraping of IEX market clearing prices.
 
Pre-processing: Handling missing time blocks and temporal encoding.


 
Predictive AI: Parallel ensemble forecasting using CNN-LSTM and XGBoost.


 
Prescriptive AI: LangGraph agent generates dispatch commands (Charge/Discharge).



# Future Roadmap

 Weather API Integration: Incorporating real-time solar irradiance to adjust charging schedules based on cloud cover.


  Calendar Awareness: Factoring in high-usage days via calendar APIs for proactive battery management.
  
  P2P Trading: Transitioning to a blockchain-based ledger for peer-to-peer energy trading between neighbors.







**Would you like me to help you write the "How to Install" section for your README based on the libraries you used?**
