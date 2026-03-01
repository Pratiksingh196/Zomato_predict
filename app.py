import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Zomato KPT Predictor", 
    page_icon="🍔", 
    layout="centered"
)

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

# --- 3. Dashboard UI ---
st.title("🍔 Zomato Prep Time Predictor")
st.markdown("""
Predict exactly how long an order will take to prepare based on real-time kitchen loads and environmental factors. 
This helps optimize rider dispatch and ensure food arrives hot!
""")

if model is None:
    st.error("⚠️ Model artifacts not found! Please ensure the three .pkl files are in the same folder as this script.")
else:
    st.header("Live Order Conditions")
    
    # Create two columns for a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kitchen Data")
        active_orders = st.number_input("Active Orders in Kitchen", min_value=0, max_value=50, value=5)
        order_items = st.number_input("Items in Current Order", min_value=1, max_value=20, value=2)
        complexity = st.slider("Order Complexity Score", min_value=1, max_value=5, value=2, help="1 = Very Easy (Drinks), 5 = Very Complex (Gourmet Meals)")
        
    with col2:
        st.subheader("Environment Data")
        time_slot = st.selectbox("Time Slot", ["Morning", "Lunch_Peak", "Tea_Time", "Dinner_Peak", "Late_Night"])
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy", "Cloudy"])
        day_type = st.selectbox("Day of Week", ["Weekday", "Weekend"])
        
    st.markdown("---")
    
    # --- 4. Prediction Logic ---
    if st.button("🚀 Predict Prep Time", type="primary", use_container_width=True):
        
        # Step A: Initialize a dictionary with ALL features the model expects, set to 0
        input_dict = {feat: 0 for feat in features}
        
        # Step B: Map the numerical inputs from the UI
        if 'Active_Orders_In_Kitchen' in input_dict: 
            input_dict['Active_Orders_In_Kitchen'] = active_orders
        if 'Order_Item_Count' in input_dict: 
            input_dict['Order_Item_Count'] = order_items
        if 'Complexity_Score' in input_dict: 
            input_dict['Complexity_Score'] = complexity
            
        # Step C: Map the categorical (One-Hot Encoded) inputs
        # The UI dropdown matches the suffix of the dummy columns we created earlier
        time_col = f"Time_Slot_{time_slot}"
        if time_col in input_dict: 
            input_dict[time_col] = 1
            
        weather_col = f"Weather_Condition_{weather}"
        if weather_col in input_dict: 
            input_dict[weather_col] = 1
            
        day_col = f"Day_of_Week_{day_type}"
        if day_col in input_dict: 
            input_dict[day_col] = 1
            
        # Provide baseline defaults for static restaurant data to avoid skewed predictions
        if 'Votes' in input_dict: input_dict['Votes'] = 200
        if 'Average Cost for two' in input_dict: input_dict['Average Cost for two'] = 500
        if 'Is delivering now' in input_dict: input_dict['Is delivering now'] = 1
        if 'Has Online delivery' in input_dict: input_dict['Has Online delivery'] = 1
        
        # Step D: Convert to DataFrame, Scale, and Predict
        input_df = pd.DataFrame([input_dict])
        
        # Use the loaded scaler to standardize the new input exactly like the training data
        input_scaled = scaler.transform(input_df)
        
        # Get the prediction
        predicted_minutes = model.predict(input_scaled)[0]
        
        # Ensure prediction doesn't drop below a logical minimum (e.g., 2 minutes)
        final_prediction = max(2.0, predicted_minutes)
        
        # --- 5. Display Result ---
        st.success(f"### ⏱️ Estimated Prep Time: {final_prediction:.1f} minutes")
        st.info(f"💡 **Dispatch Recommendation:** Assign a delivery partner who is approximately {final_prediction:.1f} minutes away from the restaurant to achieve a perfect zero-wait handover.")