from datetime import datetime

def get_tod_tariff_multiplier(timestamp: datetime) -> float:
    """
    Returns a multiplier for the base grid price based on Indian ToD (Time-of-Day) blocks.
    - 06:00 to 10:00 - Normal (1.0x)
    - 10:00 to 18:00 - Solar Cheap (0.8x) - 20% discount
    - 18:00 to 22:00 - Peak Surcharge (1.2x) - 20% surcharge
    - 22:00 to 06:00 - Normal (1.0x) - night off-peak
    """
    hour = timestamp.hour
    
    if 10 <= hour < 18:
        return 0.8  # Solar hours discount
    elif 18 <= hour < 22:
        return 1.2  # Peak hours surcharge
    else:
        return 1.0  # Normal hours

def is_solar_available(timestamp: datetime, cloud_cover_pct: float) -> bool:
    """
    Simple logic: Solar is available during the day if it's not too cloudy.
    Returns True if between 8 AM and 5 PM and cloud cover is less than 60%.
    """
    hour = timestamp.hour
    if 8 <= hour < 17 and cloud_cover_pct < 60.0:
        return True
    return False
