import math

class SpatialAgent:
    def __init__(self):
        self.earth_radius = 6371  # km

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculates distance between two points in km."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return self.earth_radius * c

    def estimate_travel_time(self, distance_km, traffic_level="Moderate"):
        """Estimates travel time based on distance and traffic."""
        avg_speed_kmh = 30  # Default 30 km/h for riders
        
        if traffic_level == "Heavy":
            avg_speed_kmh = 15
        elif traffic_level == "Clear":
            avg_speed_kmh = 45
            
        time_hours = distance_km / avg_speed_kmh
        return time_hours * 60  # minutes

    def synchronize_handover(self, kpt_mins, travel_time_mins):
        """Determines if it's the right time to dispatch."""
        wait_time = kpt_mins - travel_time_mins
        if wait_time > 2:
            return {
                "status": "Wait",
                "instruction": f"Delay dispatch by {round(wait_time, 1)} mins to ensure rider arrives exactly for handover.",
                "wait_mins": round(wait_time, 1)
            }
        elif wait_time < -5:
            return {
                "status": "Urgent",
                "instruction": "Food will be ready before rider arrives! Dispatch nearest rider immediately.",
                "wait_mins": 0
            }
        else:
            return {
                "status": "Dispatch Now",
                "instruction": "Perfect synchronization. Dispatch rider immediately for a zero-wait handover.",
                "wait_mins": 0
            }
