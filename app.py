import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# Team mappings (from training data)
teams = ['Chennai Super Kings', 'Delhi Capitals', 'Delhi Daredevils', 'Gujarat Lions',
         'Kings XI Punjab', 'Kolkata Knight Riders', 'Mumbai Indians', 'Pune Warriors',
         'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
         'Deccan Chargers', 'Kochi Tuskers Kerala', 'Rising Pune Supergiant', 'Gujarat Titans',
         'Lucknow Super Giants']

cities = ['Ahmedabad', 'Bangalore', 'Chandigarh', 'Chennai', 'Delhi', 'Dharamsala', 'Dubai',
          'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune', 'Sharjah', 'Abu Dhabi',
          'Bengaluru', 'Indore', 'Kanpur', 'Visakhapatnam', 'Rajkot']

st.title("🏏 IPL Win Probability Predictor")
st.markdown("---")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Match Details")
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

with col2:
    st.subheader("Match Situation")
    target = st.number_input("Target Score", min_value=1, max_value=300, value=180)
    current_score = st.number_input("Current Score", min_value=0, max_value=300, value=50)
    overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=19.9, value=10.0, step=0.1)
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3)

st.markdown("---")

if st.button("🎯 Predict Win Probability", type="primary"):
    if batting_team == bowling_team:
        st.error("⚠️ Batting and Bowling teams cannot be the same!")
    elif model is None:
        st.error("❌ Model not loaded. Cannot make prediction.")
    else:
        try:
            # Calculate derived features
            runs_left = target - current_score
            balls_left = 120 - (int(overs_completed) * 6 + int((overs_completed % 1) * 10))
            wickets_left = 10 - wickets_lost
            
            # Avoid division by zero
            cur_run_rate = (current_score * 6) / (int(overs_completed) * 6 + int((overs_completed % 1) * 10)) if overs_completed > 0 else 0
            req_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0
            
            # Predicted total (simple estimation)
            predicted_total = current_score + (cur_run_rate * balls_left / 6) if cur_run_rate > 0 else current_score
            
            # Prepare input: [Batting_team, Bowling_team, city, runs_left, balls_left, wickets_left, Total_x, cur_run_rate, req_run_rate, Predicted_total_run_batter]
            inputs = pd.DataFrame([[
                batting_team, bowling_team, city, runs_left, balls_left, wickets_left,
                target, cur_run_rate, req_run_rate, predicted_total
            ]], columns=['Batting_team', 'Bowling_team', 'city', 'runs_left', 'balls_left', 
                        'wickets_left', 'Total_x', 'cur_run_rate', 'req_run_rate', 'Predicted_total_run_batter'])
            
            # Make prediction
            prediction = model.predict(inputs)[0]
            probability = model.predict_proba(inputs)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            if prediction == 1:
                st.success(f"### 🎉 {batting_team} will likely WIN!")
                st.metric("Win Probability", f"{probability[1]*100:.1f}%")
            else:
                st.error(f"### 😔 {batting_team} will likely LOSE")
                st.metric("Loss Probability", f"{probability[0]*100:.1f}%")
            
            # Show match stats
            with st.expander("📈 Match Statistics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Runs Needed", runs_left)
                col2.metric("Balls Remaining", balls_left)
                col3.metric("Wickets Left", wickets_left)
                
                col1, col2 = st.columns(2)
                col1.metric("Current Run Rate", f"{cur_run_rate:.2f}")
                col2.metric("Required Run Rate", f"{req_run_rate:.2f}")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check if all inputs are valid.")
