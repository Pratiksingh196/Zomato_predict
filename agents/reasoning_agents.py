import ollama
import json

class ReasoningAgents:
    def __init__(self, model="gpt-oss:120b-cloud"):
        self.model = model

    def _query_llm(self, prompt):
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

    def forecaster_agent(self, recent_load, environmental_factors):
        """Predicts future kitchen load based on current trends."""
        prompt = f"""
        Current Kitchen Load: {recent_load} orders.
        Environment: {environmental_factors}.
        
        Task: Predict the kitchen load for the next 30 minutes.
        Requirements:
        1. Use extremely simple, easy-to-understand language.
        2. Provide exactly 2-3 concise bullet points.
        3. Do NOT use JSON. Use Markdown bullet points (*).
        4. Focus only on what the manager needs to know.
        """
        return self._query_llm(prompt)

    def dispatch_agent(self, predicted_kpt, rider_dist, traffic_level, spatial_context=""):
        """Decides the optimal dispatch timing with spatial context."""
        prompt = f"""
        Food Prep Time: {predicted_kpt} mins.
        Rider Distance: {rider_dist} km.
        Traffic: {traffic_level}.
        
        Task: Advice on rider dispatch timing.
        Requirements:
        1. Use very simple language.
        2. Provide exactly 2-3 concise bullet points. 
        3. Do NOT use JSON. Use Markdown bullet points (*).
        4. Tell the manager exactly when to call or release the rider.
        """
        return self._query_llm(prompt)

    def learning_agent(self, predicted, actual, context):
        """Analyzes prediction errors and provides insights."""
        error = actual - predicted
        prompt = f"""
        Prediction: {predicted} mins | Actual: {actual} mins | Gap: {error} mins.
        Context: {context}.
        
        Task: Why was there a gap?
        Requirements:
        1. Use simple, non-technical language.
        2. Provide exactly 2 concise bullet points.
        3. Do NOT use JSON. Use Markdown bullet points (*).
        4. Give one 'Lesson Learned' for future orders.
        """
        return self._query_llm(prompt)

    def explainer_agent(self, question, state_data):
        """Natural language interface for kitchen managers."""
        prompt = f"""
        Manager Question: "{question}"
        Current State: {state_data}
        Answer as a helpful RasoiSetu AI. Keep it short, simple, and professional.
        Use bullet points if listing things.
        """
        return self._query_llm(prompt)

    def anomaly_agent(self, current_stats):
        """Scans for systemic failures or abnormal spikes."""
        prompt = f"""
        Current Stats: {current_stats}
        
        Task: Is there a problem?
        Requirements:
        1. Use simple language.
        2. Provide exactly 2 concise bullet points.
        3. Do NOT use JSON. Use Markdown bullet points (*).
        4. If everything is normal, say "System Healthy" and why.
        """
        return self._query_llm(prompt)
