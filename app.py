import streamlit as st
import pickle
import pandas as pd

# Load models
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading Random Forest model: {e}")
    model = None

# Neural network model disabled for deployment
neural_model = None

# Load player data
try:
    df = pd.read_excel('ui_player_data.xlsx')
    player_data_dict = {row['Player Name']: [row['Total Not Outs'], row['Highest Score'], row['Strike Rate']] 
                        for _, row in df.iterrows()}
except Exception as e:
    st.warning(f"Could not load player data: {e}")
    player_data_dict = {}

# Team and city mappings
batting_team = {'Royal Challengers Bangalore': 0, 'Rising Pune Supergiant': 1, 'Kolkata Knight Riders': 2, 
                'Kings XI Punjab': 3, 'Delhi Daredevils': 4, 'Sunrisers Hyderabad': 5, 'Mumbai Indians': 6, 
                'Gujarat Lions': 7, 'Chennai Super Kings': 8, 'Rajasthan Royals': 9, 'Delhi Capitals': 10, 
                'Deccan Chargers': 11}

bowling_team = {'Sunrisers Hyderabad': 0, 'Mumbai Indians': 1, 'Gujarat Lions': 2, 'Rising Pune Supergiant': 3, 
                'Royal Challengers Bangalore': 4, 'Kolkata Knight Riders': 5, 'Delhi Daredevils': 6, 
                'Kings XI Punjab': 7, 'Rajasthan Royals': 8, 'Chennai Super Kings': 9, 'Delhi Capitals': 10, 
                'Deccan Chargers': 11}

city = {'Hyderabad': 0, 'Pune': 1, 'Rajkot': 2, 'Indore': 3, 'Bengaluru': 4, 'Mumbai': 5, 'Kolkata': 6, 
        'Bangalore': 7, 'Delhi': 8, 'Chandigarh': 9, 'Kanpur': 10, 'Chennai': 11, 'Jaipur': 12, 
        'Visakhapatnam': 13, 'Abu Dhabi': 14, 'Dubai': 15, 'UAE': 16, 'Ahmedabad': 17, 'Sharjah': 18, 
        'Navi Mumbai': 19, 'Guwahati': 20, 'Cape Town': 21, 'Port Elizabeth': 22, 'Durban': 23, 
        'Centurion': 24, 'East London': 25, 'Johannesburg': 26, 'Kimberley': 27, 'Bloemfontein': 28, 
        'Cuttack': 29, 'Nagpur': 30, 'Dharamsala': 31, 'Raipur': 32, 'Ranchi': 33}

st.title("IPL Win Probability Predictor")

# Show model info
if model is not None:
    try:
        st.sidebar.info(f"Model expects {model.n_features_in_} features")
    except:
        pass

# Input fields
batting = st.selectbox("Batting Team", list(batting_team.keys()))
bowling = st.selectbox("Bowling Team", list(bowling_team.keys()))
city_name = st.selectbox("City", list(city.keys()))
runs = st.number_input("Current Runs", min_value=0, value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=0)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
target = st.number_input("Target", min_value=0, value=0)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Cannot make prediction.")
    else:
        try:
            # Prepare input - adjust based on your model's expected features
            import numpy as np
            inputs = np.array([[batting_team[batting], bowling_team[bowling], city[city_name], 
                       runs, wickets, overs, target, 0, 0, 0]], dtype=float)
            
            prediction = model.predict(inputs)[0]
            
            if prediction == 0:
                st.error(f"🔴 {batting} will likely Lose")
            else:
                st.success(f"🟢 {batting} will likely Win")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("The model expects specific input features. Please check the model training code.")
