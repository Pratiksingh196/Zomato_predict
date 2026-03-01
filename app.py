import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import random
from agents.data_agent import DataAgent
from agents.reasoning_agents import ReasoningAgents
from agents.weather_agent import WeatherAgent
from agents.spatial_agent import SpatialAgent

# Initialize Agents
data_agent = DataAgent()
reasoning_agents = ReasoningAgents()
weather_agent = WeatherAgent()
spatial_agent = SpatialAgent()

# Helper: Map Current Time to Slot
def get_current_time_slot():
    hour = time.localtime().tm_hour
    if 5 <= hour < 11: return "Morning"
    elif 11 <= hour < 15: return "Lunch_Peak"
    elif 15 <= hour < 18: return "Tea_Time"
    elif 18 <= hour < 22: return "Dinner_Peak"
    else: return "Late_Night"

def get_day_type():
    wday = time.localtime().tm_wday
    return "Weekend" if wday >= 5 else "Weekday"

# --- 1. Page Configuration ---
st.set_page_config(page_title="RasoiSetu | Agentic AI", page_icon="🍔", layout="wide")

# --- 2. Load Model Artifacts ---
# We use @st.cache_resource so the model loads only once when the app starts, making it lightning fast.
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('zomato_kpt_ridge_model.pkl')
        scaler = joblib.load('zomato_kpt_scaler.pkl')
        features = joblib.load('kpt_model_features.pkl')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

model, scaler, features = load_artifacts()

# --- 3. Sidebar & Navigation ---
# sidebar.title uses markdown to support Font Awesome
st.sidebar.markdown('<h1><i class="fa-solid fa-microchip"></i> RasoiSetu</h1>', unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Mission Control", ["Live Predictor", "Crisis Scenario AI", "Kitchen AI Chatbot"])

if "auto_context" not in st.session_state:
    with st.spinner("🤖 Agents auto-detecting context..."):
        loc_info = weather_agent.auto_locate() # Returns {city, lat, lon}
        weather_data = weather_agent.get_weather(loc_info["city"])
        curr_slot = get_current_time_slot()
        curr_day = get_day_type()
        
        # Default Locations: Pitampura & Shalimar Bagh, Delhi
        st.session_state.auto_context = {
            "city": "Pitampura, Delhi",
            "lat": 28.6542,
            "lon": 77.2373,
            "rider_city": "Shalimar Bagh, Delhi",
            "rider_lat": 28.6642,
            "rider_lon": 77.2473,
            "weather": weather_data,
            "time_slot": curr_slot,
            "day_type": curr_day
        }

if "predictor_inputs" not in st.session_state:
    # Initialize from auto-context
    ctx = st.session_state.auto_context
    st.session_state.predictor_inputs = {
        "active_orders": 5, "num_items": 2, "complexity": 3,
        "weather": ctx["weather"].get("mapped_condition", "Clear"),
        "time_slot": ctx["time_slot"], "day_type": ctx["day_type"], "traffic": "Moderate"
    }

if app_mode == "Live Predictor":
    st.markdown('<h1 class="icon-title"><i class="fa-solid fa-fire-flame-curved" style="color:#e23744"></i> RasoiSetu Control Center</h1>', unsafe_allow_html=True)
    st.markdown("Precision KPT prediction powered by a distributed multi-agent system.")
    
    if model is None:
        st.error("⚠️ Model artifacts not found! Please ensure the three .pkl files are in the same folder as this script.")
    else:
        # --- 3. Input Section ---
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<h3><i class="fa-solid fa-kitchen-set"></i> Kitchen Real-time Data</h3>', unsafe_allow_html=True)
            active_orders = st.number_input("Active Orders in Kitchen", min_value=1, value=st.session_state.predictor_inputs["active_orders"])
            num_items = st.number_input("Items in Current Order", min_value=1, value=st.session_state.predictor_inputs["num_items"])
            complexity = st.slider("Order Complexity Score", 1, 5, int(st.session_state.predictor_inputs["complexity"]), help="1 = Very Easy (Drinks), 5 = Very Complex (Gourmet Meals)")
            
        with col2:
            st.markdown('<h3><i class="fa-solid fa-cloud-sun"></i> Environmental Context</h3>', unsafe_allow_html=True)
            
            # Use state for defaults
            ctx = st.session_state.auto_context
            city_input = st.text_input("Current City", value=ctx["city"])
            
            weather_list = ["Clear", "Rainy", "Stormy", "Cloudy"]
            slot_list = ["Morning", "Lunch_Peak", "Tea_Time", "Dinner_Peak", "Late_Night"]
            
            s_weather = st.session_state.predictor_inputs["weather"]
            s_slot = st.session_state.predictor_inputs["time_slot"]
            s_day = st.session_state.predictor_inputs["day_type"]
            s_traffic = st.session_state.predictor_inputs["traffic"]
            
            weather = st.selectbox("Weather Condition", weather_list, index=weather_list.index(s_weather) if s_weather in weather_list else 0)
            time_slot = st.selectbox("Time Slot", slot_list, index=slot_list.index(s_slot) if s_slot in slot_list else 0)
            day_type = st.selectbox("Day of Week", ["Weekday", "Weekend"], index=1 if s_day == "Weekend" else 0)
            traffic = st.select_slider("Traffic Density", options=["Clear", "Moderate", "Heavy"], value=s_traffic)
            
            if st.button("Sync Live Environment"):
                with st.spinner("Refetching context..."):
                    new_coords = weather_agent.get_city_coords(city_input)
                    new_weather = weather_agent.get_weather(city_input)
                    st.session_state.auto_context.update({
                        "weather": new_weather, "city": city_input, "lat": new_coords["lat"], "lon": new_coords["lon"],
                        "rider_lat": new_coords["lat"] + 0.015, "rider_lon": new_coords["lon"] + 0.015
                    })
                    st.rerun()
                    
    st.markdown("---")
    st.markdown('<h3><i class="fa-solid fa-location-dot"></i> Spatial Intelligence Hub</h3>', unsafe_allow_html=True)
    
    # --- Spatial Inputs ---
    ctx = st.session_state.auto_context
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown('**<i class="fa-solid fa-shop"></i> Restaurant Node**', unsafe_allow_html=True)
        k_city = st.text_input("Origin City", value=ctx.get("city", "Pitampura, Delhi"), key="k_city_search")
        if st.button("Resolve Origin"):
            with st.spinner("Geocoding..."):
                coords = weather_agent.get_city_coords(k_city)
                st.session_state.auto_context.update({"city": k_city, "lat": coords["lat"], "lon": coords["lon"]})
                st.rerun()
        
        k_lat = st.number_input("Latitude", value=float(ctx["lat"]), format="%.4f", key="k_lat_input")
        k_lon = st.number_input("Longitude", value=float(ctx["lon"]), format="%.4f", key="k_lon_input")
        
    with col_s2:
        st.markdown('**<i class="fa-solid fa-person-biking"></i> Delivery Partner Node**', unsafe_allow_html=True)
        r_city = st.text_input("Partner City", value=ctx.get("rider_city", "Shalimar Bagh, Delhi"), key="r_city_search")
        if st.button("Resolve Partner"):
            with st.spinner("Geocoding..."):
                coords = weather_agent.get_city_coords(r_city)
                st.session_state.auto_context.update({"rider_city": r_city, "rider_lat": coords["lat"], "rider_lon": coords["lon"]})
                st.rerun()
                
        r_lat = st.number_input("Latitude", value=float(ctx["rider_lat"]), format="%.4f", key="r_lat_input")
        r_lon = st.number_input("Longitude", value=float(ctx["rider_lon"]), format="%.4f", key="r_lon_input")

    if st.button("Calibrate Spatial Systems", use_container_width=True):
        with st.spinner("Calibrating..."):
            st.session_state.auto_context.update({"lat": k_lat, "lon": k_lon, "rider_lat": r_lat, "rider_lon": r_lon})
            st.rerun()

    # Map Visualization
    ctx = st.session_state.auto_context
    map_data = pd.DataFrame({'lat': [ctx["lat"], ctx["rider_lat"]], 'lon': [ctx["lon"], ctx["rider_lon"]], 'name': ['Origin', 'Partner']})
    st.map(map_data, zoom=6 if ctx.get("city") != ctx.get("rider_city") else 12)
    
    distance = spatial_agent.haversine(ctx["lat"], ctx["lon"], ctx["rider_lat"], ctx["rider_lon"])
    travel_time = spatial_agent.estimate_travel_time(distance, traffic)
    
    st.markdown(f"""
    <div style="background:#f3f4f6; color:#1f2937; padding:15px; border-radius:10px; border:1px solid #e5e7eb">
        <i class="fa-solid fa-route"></i> <b>Distance:</b> {distance:.2f} km | 
        <i class="fa-solid fa-clock"></i> <b>Transit Time:</b> {travel_time:.1f} mins ({traffic})
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- 4. Prediction Logic ---
    if st.button("Run Operational Prediction", type="primary", use_container_width=True):
        # Sync state
        st.session_state.predictor_inputs.update({
            "active_orders": active_orders, "num_items": num_items, "complexity": complexity,
            "weather": weather, "time_slot": time_slot, "day_type": day_type, "traffic": traffic
        })
        
        input_dict = {str(feat): 0 for feat in features}
        if 'Active_Orders_In_Kitchen' in input_dict: input_dict['Active_Orders_In_Kitchen'] = active_orders
        if 'Order_Item_Count' in input_dict: input_dict['Order_Item_Count'] = num_items
        if 'Complexity_Score' in input_dict: input_dict['Complexity_Score'] = complexity
            
        for key, val in [(f"Time_Slot_{time_slot}", 1), (f"Weather_Condition_{weather}", 1), (f"Day_of_Week_{day_type}", 1)]:
            if key in input_dict: input_dict[key] = val
            
        for k in ['Votes', 'Average Cost for two', 'Is delivering now', 'Has Online delivery']:
            if k in input_dict: input_dict[k] = 500 if 'Cost' in k else (200 if 'Votes' in k else 1)
        
        input_df = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)
        predicted_minutes = model.predict(input_scaled)[0]
        final_prediction = max(2.0, predicted_minutes)
        st.session_state.last_prediction = final_prediction
        
        st.success(f"### ⏱️ Estimated Prep Time: {final_prediction:.1f} minutes")
        
        st.markdown("---")
        st.markdown('<h3><i class="fa-solid fa-clipboard-check"></i> Multi-Agent Executive Audit</h3>', unsafe_allow_html=True)
        context_str = f"Orders: {active_orders}, Slot: {time_slot}, Weather: {weather}, Day: {day_type}"
        
        grid1, grid2 = st.columns(2)
        with grid1:
            st.markdown('<div class="report-card"><h4><i class="fa-solid fa-truck-ramp-box"></i> Spatial Handover Agent</h4>', unsafe_allow_html=True)
            with st.spinner("Syncing..."):
                dispatch_advice = reasoning_agents.dispatch_agent(final_prediction, distance, traffic, "")
                st.markdown(dispatch_advice, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-card" style="border-left-color:#3b82f6"><h4><i class="fa-solid fa-chart-line"></i> Predictive Forecaster</h4>', unsafe_allow_html=True)
            with st.spinner("Projecting..."):
                forecast = reasoning_agents.forecaster_agent(active_orders, f"{weather}, {time_slot}")
                st.markdown(forecast)
            st.markdown('</div>', unsafe_allow_html=True)

        with grid2:
            st.markdown('<div class="report-card" style="border-left-color:#f59e0b"><h4><i class="fa-solid fa-triangle-exclamation"></i> System Health Agent</h4>', unsafe_allow_html=True)
            with st.spinner("Scanning..."):
                anomaly_report = reasoning_agents.anomaly_agent({"kpt": final_prediction, "load": active_orders})
                st.markdown(anomaly_report)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-card" style="border-left-color:#10b981"><h4><i class="fa-solid fa-brain"></i> Optimization Brain</h4>', unsafe_allow_html=True)
            with st.spinner("Self-Correcting..."):
                insight = reasoning_agents.learning_agent(final_prediction, final_prediction*1.05, context_str)
                st.markdown(insight)
            st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Crisis Scenario AI":
    st.markdown('<h1 class="icon-title"><i class="fa-solid fa-cloud-bolt" style="color:#3b82f6"></i> Crisis Scenario Generator</h1>', unsafe_allow_html=True)
    st.markdown("Simulate extreme edge cases to test system resilience.")
    
    event_type = st.selectbox("Trigger Event", ["Monsoon Cloudburst", "IPL Final Match", "Festival (Diwali/Holi)", "Public Strike"])
    
    if st.button("Synthesize Scenario", type="primary"):
        with st.spinner("Data Agent generating edge-case JSON..."):
            scenario = data_agent.generate_scenario(event_type)
            st.session_state.last_scenario = scenario # For injection
            st.markdown('<div class="report-card" style="border-left-color:#3b82f6">', unsafe_allow_html=True)
            st.json(scenario)
            st.markdown('</div>', unsafe_allow_html=True)

    if "last_scenario" in st.session_state:
        st.write("---")
        if st.button("� LOAD SCENARIO INTO PREDICTOR", use_container_width=True):
            s = st.session_state.last_scenario
            st.session_state.predictor_inputs.update({
                "active_orders": s.get("active_orders", 5),
                "num_items": random.randint(2, 6),
                "complexity": s.get("avg_complexity", 3),
                "weather": s.get("weather", "Clear"),
                "time_slot": s.get("time_slot", "Morning"),
                "day_type": s.get("day_type", "Weekday"),
                "traffic": "Heavy" if s.get("weather") in ["Rainy", "Stormy"] else "Moderate"
            })
            st.success("Target scenario loaded! Switch to **Live Predictor** missions.")
            st.info("💡 Predictor dials are now pre-filled with the Crisis Data.")

elif app_mode == "Kitchen AI Chatbot":
    st.markdown('<h1 class="icon-title"><i class="fa-solid fa-comments" style="color:#10b981"></i> Kitchen AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask the Explainer Agent about current operations or prediction logic.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help with the shift today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting reasoning nodes..."):
                state_summary = f"Orders: {st.session_state.get('auto_context', {}).get('city', 'Unknown')}"
                response = reasoning_agents.explainer_agent(prompt, state_summary)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})