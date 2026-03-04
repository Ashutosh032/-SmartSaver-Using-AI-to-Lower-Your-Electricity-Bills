class BatteryEnvironment:
    def __init__(self, capacity_kwh=10.0, max_charge_rate_kw=5.0, initial_soc=50.0):
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate_kw = max_charge_rate_kw
        self.soc = initial_soc # State of Charge %
        self.interval_hours = 0.25 # 15 minutes
        
    def get_soc(self):
        return self.soc
        
    def step(self, action: str, load_kw: float = 2.0, solar_kw: float = 3.0):
        """
        Updates battery state and calculates energy cost from grid.
        action: CHARGE_GRID, CHARGE_SOLAR, DISCHARGE, IDLE
        load_kw: home usage
        solar_kw: solar generated
        Returns: grid_energy_kwh (positive means bought from grid, negative means sold/not used)
        """
        grid_energy_kwh = 0.0
        
        # Max energy we can move in one interval
        max_energy_kwh = self.max_charge_rate_kw * self.interval_hours
        
        # Current energy in battery
        current_energy_kwh = (self.soc / 100.0) * self.capacity_kwh
        
        load_energy_kwh = load_kw * self.interval_hours
        solar_energy_kwh = solar_kw * self.interval_hours
        
        if action == "CHARGE_SOLAR":
            # Use solar to power home and charge battery
            remaining_solar = max(0, solar_energy_kwh - load_energy_kwh)
            charge_amount = min(remaining_solar, max_energy_kwh, self.capacity_kwh - current_energy_kwh)
            current_energy_kwh += charge_amount
            # Grid needs to supply any load solar couldn't cover
            grid_energy_kwh = max(0, load_energy_kwh - solar_energy_kwh)
            
        elif action == "CHARGE_GRID":
            # Charge from Grid while grid also powers the home
            charge_amount = min(max_energy_kwh, self.capacity_kwh - current_energy_kwh)
            current_energy_kwh += charge_amount
            grid_energy_kwh = load_energy_kwh + charge_amount
            
        elif action == "DISCHARGE":
            # Use battery to cover home load
            discharge_amount = min(load_energy_kwh, max_energy_kwh, current_energy_kwh - (0.2 * self.capacity_kwh))
            # 0.2 * capacity is the absolute minimum 20% limit enforced by physics here just in case
            if discharge_amount < 0: discharge_amount = 0
            current_energy_kwh -= discharge_amount
            grid_energy_kwh = load_energy_kwh - discharge_amount
            
        elif action == "IDLE":
            # Battery does nothing, grid powers home, solar might offset
            grid_energy_kwh = max(0, load_energy_kwh - solar_energy_kwh)
            
        # Update SOC
        self.soc = (current_energy_kwh / self.capacity_kwh) * 100.0
        
        return grid_energy_kwh
