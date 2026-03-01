import ollama
import pandas as pd
import json
import random

class DataAgent:
    def __init__(self, model="gpt-oss:120b-cloud"):
        self.model = model

    def generate_scenario(self, scenario_type="Festival"):
        """Generates a synthetic kitchen scenario based on a specific type."""
        prompt = f"""
        Generate a realistic Zomato kitchen scenario for a '{scenario_type}' event in an Indian city.
        Return ONLY a JSON object with the following fields:
        - scenario_name: String
        - active_orders: Integer (10-50)
        - avg_complexity: Float (1.0-5.0)
        - time_slot: String (Morning, Lunch_Peak, Tea_Time, Dinner_Peak, Late_Night)
        - weather: String (Clear, Rainy, Stormy, Cloudy)
        - day_type: String (Weekday, Weekend)
        - description: String (short context)
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            # Assuming the response content is a clean JSON string
            data = json.loads(response['message']['content'])
            return data
        except Exception as e:
            # Fallback data if LLM fails or model not found
            return {
                "scenario_name": f"Mock {scenario_type} Event",
                "active_orders": random.randint(15, 45),
                "avg_complexity": round(random.uniform(2.5, 4.8), 1),
                "time_slot": "Dinner_Peak",
                "weather": "Rainy",
                "day_type": "Weekend",
                "description": "High volume due to simulated event bottleneck."
            }

    def augment_dataset(self, num_records=10):
        """Generates multiple records to simulate a dataset expansion."""
        scenarios = ["IPL Match", "Heavy Monsoon", "Diwali Night", "Sunday Brunch Rush"]
        new_data = []
        for _ in range(num_records):
            s_type = random.choice(scenarios)
            new_data.append(self.generate_scenario(s_type))
        
        return pd.DataFrame(new_data)

if __name__ == "__main__":
    agent = DataAgent()
    print("Generating Sample Scenario...")
    print(agent.generate_scenario("IPL Match"))
