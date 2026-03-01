import requests

class WeatherAgent:
    def __init__(self, api_key="f00c38e0279b7bc85480c3fe775d518c"):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def auto_locate(self):
        """Detects the current city and coordinates via IP-based geolocation."""
        try:
            response = requests.get("http://ip-api.com/json")
            data = response.json()
            if data.get("status") == "success":
                return {
                    "city": data.get("city", "Pune"),
                    "lat": data.get("lat", 18.5204),
                    "lon": data.get("lon", 73.8567)
                }
            return {"city": "Pune", "lat": 18.5204, "lon": 73.8567}
        except:
            return {"city": "Pune", "lat": 18.5204, "lon": 73.8567}

    def get_city_coords(self, city):
        """Fetches coordinates for a given city (Fallback to Pune)."""
        params = {"q": city, "appid": self.api_key, "limit": 1}
        try:
            url = "http://api.openweathermap.org/geo/1.0/direct"
            resp = requests.get(url, params=params)
            data = resp.json()
            if data:
                return {"lat": data[0]['lat'], "lon": data[0]['lon']}
            return {"lat": 18.5204, "lon": 73.8567}
        except:
            return {"lat": 18.5204, "lon": 73.8567}

    def get_weather(self, city="Pune"):
        """Fetches live weather and maps it to model categories."""
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            if response.status_code == 200:
                condition = data['weather'][0]['main']
                temp = data['main']['temp']
                
                # Mapping OpenWeatherMap conditions to Model categories
                # Categories: ["Clear", "Rainy", "Stormy", "Cloudy"]
                mapped_condition = "Clear"
                if condition in ["Rain", "Drizzle"]:
                    mapped_condition = "Rainy"
                elif condition == "Thunderstorm":
                    mapped_condition = "Stormy"
                elif condition in ["Clouds", "Fog", "Mist", "Haze"]:
                    mapped_condition = "Cloudy"
                
                return {
                    "city": data['name'],
                    "temp": temp,
                    "condition": condition,
                    "mapped_condition": mapped_condition,
                    "icon": data['weather'][0]['icon']
                }
            else:
                return {"error": data.get("message", "City not found")}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    agent = WeatherAgent()
    print(agent.get_weather("Pune"))
