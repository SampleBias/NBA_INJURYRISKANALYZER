import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PROCESSED_DATA_PATH = 'nba_processed_data.pkl'
MODEL_PATH = 'nba_model.joblib'
SCALER_PATH = 'nba_scaler.joblib'

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            return pd.read_pickle(PROCESSED_DATA_PATH)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.error(f"Processed data not found at {PROCESSED_DATA_PATH}")
        return None

@st.cache_resource
def load_model() -> Tuple[Optional[object], Optional[object]]:
    model, scaler = None, None
    for path, obj_name in [(MODEL_PATH, "model"), (SCALER_PATH, "scaler")]:
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                if obj_name == "model":
                    model = obj
                else:
                    scaler = obj
            except Exception as e:
                st.error(f"Error loading {obj_name}: {e}")
        else:
            st.error(f"{obj_name.capitalize()} not found at {path}")
    return model, scaler

def predict_injury_risk(model, scaler, df_processed, input_data):
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make prediction.")
        return None, None

    new_data = pd.DataFrame([input_data])
    feature_names = df_processed.drop(['Injury_Risk', 'PLAYER_NAME'], axis=1).columns
    new_data = new_data.reindex(columns=feature_names, fill_value=0)
    X_new_scaled = scaler.transform(new_data)
    risk_probability = model.predict_proba(X_new_scaled)[0][1]
    risk_prediction = model.predict(X_new_scaled)[0]
    return risk_prediction, risk_probability

def predict_injury_risk_for_all(model, scaler, df_processed):
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make predictions.")
        return None

    X = df_processed.drop(['Injury_Risk', 'PLAYER_NAME'], axis=1)
    X_scaled = scaler.transform(X)
    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    return pd.DataFrame({
        'PLAYER_NAME': df_processed['PLAYER_NAME'],
        'Injury_Risk_Prediction': y_pred,
        'Injury_Risk_Probability': y_probs
    })

def plot_feature_importance(model, feature_names, top_n=15):
    if model is None:
        st.error("Model not available. Unable to plot feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    st.pyplot(fig)

def display_player_stats(player_data):
    st.subheader("Player Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Age:** {player_data['AGE'].iloc[0]:.0f}")
        st.markdown(f"**Height:** {player_data['HEIGHT'].iloc[0]:.0f} inches")
        st.markdown(f"**Weight:** {player_data['WEIGHT'].iloc[0]:.0f} lbs")
        st.markdown(f"**Years of Experience:** {player_data['SEASON_EXP'].iloc[0]:.0f}")
    
    with col2:
        st.markdown("**Performance (Season Averages):**")
        for stat, label in [('PTS', 'Points'), ('AST', 'Assists'), ('REB', 'Rebounds')]:
            st.metric(label, f"{player_data[stat].mean():.1f}")
        
        for stat, label in [('FG_PCT', 'FG%'), ('FG3_PCT', '3P%'), ('FT_PCT', 'FT%')]:
            st.metric(label, f"{player_data[stat].mean():.3f}")
    
    position_columns = player_data.filter(like='POSITION_').columns
    position = position_columns[player_data[position_columns].iloc[0] == 1].str.replace('POSITION_', '').values
    st.markdown(f"**Position:** {position[0] if len(position) > 0 else 'Unknown'}")

def get_auragens_recommendation(injury_risk_pred):
    if injury_risk_pred == 1:
        return ("Auragens Treatment Recommendation: "
                "IV infusion of 150 Million Mesenchymal Stem Cells (MSCs) systemically, "
                "combined with a local injection of 5-10 Million MSCs "
                "at the site of injury, administered over 3 consecutive days. "
                "This protocol aims to promote tissue repair, modulate inflammation, "
                "and enhance recovery and improve athletic performance.")
    else:
        return ("Auragens Treatment Recommendation: "
                "Performance-enhancing injury prevention treatment consisting of "
                "intravenous (IV) infusion of 150 Million Mesenchymal Stem Cells (MSCs) "
                "administered over 3 consecutive days. "
                "This proactive protocol aims to bolster the body's natural repair mechanisms, "
                "potentially reducing injury risk and optimizing athletic performance.")

def add_player_record(df_processed, new_player_data):
    new_player_df = pd.DataFrame([new_player_data])
    new_player_df = new_player_df.reindex(columns=df_processed.columns, fill_value=0)
    return pd.concat([df_processed, new_player_df], ignore_index=True)

def main():
    st.set_page_config(page_title="NBA Injury Risk Prediction App", page_icon="🏀")
    st.title("NBA Injury Risk Prediction App 🏀")
    
    df_processed = load_data()
    model, scaler = load_model()
    
    if df_processed is None:
        st.warning("Data not available. Some features may be limited.")
    if model is None or scaler is None:
        st.warning("Model or scaler not available. Predictions cannot be made.")
    
    menu = ["Home", "Predict Injury Risk", "Player Risk Lookup", "Add Player Record", "Data Visualization", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to the NBA Injury Risk Prediction App")
        st.write("""
        This app uses machine learning to predict the injury risk for NBA players based on their statistics and historical data. 
        Use the sidebar to navigate through different features of the app:
        
        - **Predict Injury Risk**: Input player stats to get an injury risk prediction
        - **Player Risk Lookup**: Select a player to see their injury risk, stats, and treatment recommendations
        - **Add Player Record**: Add a new player to the database
        - **Data Visualization**: Explore feature importance and SHAP values
        - **About**: Learn more about how this app works
        
        Get started by selecting an option from the sidebar!
        """)
    
    elif choice == "Predict Injury Risk":
        st.subheader("Predict Injury Risk for a Player")
        
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to make predictions.")
            return

        with st.form("player_form"):
            player_name = st.text_input("Player Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=50, value=25)
                height = st.number_input("Height (inches)", min_value=60, max_value=90, value=72)
                weight = st.number_input("Weight (lbs)", min_value=150, max_value=400, value=220)
                years_exp = st.number_input("Years of Experience", min_value=0, max_value=30, value=3)
                position = st.selectbox("Position", df_processed.filter(like='POSITION_').columns.str.replace('POSITION_', ''))
            with col2:
                pts = st.number_input("Points per Game", min_value=0.0, value=10.0)
                ast = st.number_input("Assists per Game", min_value=0.0, value=2.0)
                reb = st.number_input("Rebounds per Game", min_value=0.0, value=5.0)
                fg_pct = st.number_input("Field Goal Percentage", min_value=0.0, max_value=1.0, value=0.45)
                fg3_pct = st.number_input("3-Point Percentage", min_value=0.0, max_value=1.0, value=0.35)
                ft_pct = st.number_input("Free Throw Percentage", min_value=0.0, max_value=1.0, value=0.75)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'PLAYER_NAME': player_name,
                'AGE': age,
                'HEIGHT': height,
                'WEIGHT': weight,
                'SEASON_EXP': years_exp,
                'PTS': pts,
                'AST': ast,
                'REB': reb,
                'FG_PCT': fg_pct,
                'FG3_PCT': fg3_pct,
                'FT_PCT': ft_pct,
                **{f'POSITION_{position}': 1}
            }
            
            risk_prediction, risk_probability = predict_injury_risk(model, scaler, df_processed, input_data)
            if risk_prediction is not None and risk_probability is not None:
                risk_label = 'High Risk' if risk_prediction == 1 else 'Low Risk'
                st.markdown(f"### Injury Risk Prediction for {player_name}: {risk_label}")
                st.markdown(f"#### Risk Probability: {risk_probability:.2f}")
                
                recommendation = get_auragens_recommendation(risk_prediction)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.error("Unable to make prediction. Please check the input data and try again.")
    
    elif choice == "Player Risk Lookup":
        st.subheader("NBA Players Injury Risk")
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to perform player risk lookup.")
            return

        df_results = predict_injury_risk_for_all(model, scaler, df_processed)
        if df_results is not None:
            player_names = df_processed['PLAYER_NAME'].unique()
            selected_player = st.selectbox("Select a Player", sorted(player_names))
            player_data = df_processed[df_processed['PLAYER_NAME'] == selected_player]
            if not player_data.empty:
                player_result = df_results[df_results['PLAYER_NAME'] == selected_player].iloc[0]
                injury_risk_prob = player_result['Injury_Risk_Probability']
                injury_risk_pred = player_result['Injury_Risk_Prediction']
                risk_label = 'High Risk' if injury_risk_pred == 1 else 'Low Risk'
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"<h2 style='font-size: 24px;'>{selected_player}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Injury Risk Prediction: <strong>{risk_label}</strong></h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Risk Probability: <strong>{injury_risk_prob:.2f}</strong></h3>", unsafe_allow_html=True)
                
                with col2:
                    display_player_stats(player_data)
                
                recommendation = get_auragens_recommendation(injury_risk_pred)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.write("Player data not found.")
        else:
            st.error("Unable to generate risk predictions for players.")

    elif choice == "Add Player Record":
        st.subheader("Add New Player Record")
        if df_processed is None:
            st.error("Data not available. Unable to add new player record.")
            return

        with st.form("add_player_form"):
            player_name = st.text_input("Player Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=50, value=25)
                height = st.number_input("Height (inches)", min_value=60, max_value=90, value=72)
                weight = st.number_input("Weight (lbs)", min_value=150, max_value=400, value=220)
                years_exp = st.number_input("Years of Experience", min_value=0, max_value=30, value=3)
                position = st.selectbox("Position", df_processed.filter(like='POSITION_').columns.str.replace('POSITION_', ''))
            with col2:
                pts = st.number_input("Points per Game", min_value=0.0, value=10.0)
                ast = st.number_input("Assists per Game", min_value=0.0, value=2.0)
                reb = st.number_input("Rebounds per Game", min_value=0.0, value=5.0)
                fg_pct = st.number_input("Field Goal Percentage", min_value=0.0, max_value=1.0, value=0.45)
                fg3_pct = st.number_input("3-Point Percentage", min_value=0.0, max_value=1.0, value=0.35)
                ft_pct = st.number_input("Free Throw Percentage", min_value=0.0, max_value=1.0, value=0.75)
            
            submitted = st.form_submit_button("Add Player")

        if submitted:
            new_player_data = {
                'PLAYER_NAME': player_name,
                'AGE': age,
                'HEIGHT': height,
                'WEIGHT': weight,
                'SEASON_EXP': years_exp,
                'PTS': pts,
                'AST': ast,
                'REB': reb,
                'FG_PCT': fg_pct,
                'FG3_PCT': fg3_pct,
                'FT_PCT': ft_pct,
                **{f'POSITION_{pos}': 1 if pos == position else 0 for pos in df_processed.filter(like='POSITION_').columns.str.replace('POSITION_', '')}
            }
            
            updated_df = add_player_record(df_processed, new_player_data)
            
            # Save the updated DataFrame
            updated_df.to_pickle(PROCESSED_DATA_PATH)
            st.success(f"Player {player_name} has been added successfully!")
            st.info("Please refresh the app to see the updated player list in the Player Risk Lookup section.")

    elif choice == "Data Visualization":
        st.subheader("Data Visualization")
        
        if df_processed is None or model is None or scaler is None:
            st.error("Required components are missing. Unable to generate visualizations.")
            return

        st.write("""
        This section provides insights into how our model makes predictions. We use two main visualization techniques:
        Feature Importance and SHAP (SHapley Additive exPlanations) values.
        """)
        
        st.subheader("Feature Importance")
        st.write("""
        Feature importance shows how much each feature contributes to the model's predictions. 
        Features with higher importance have a greater impact on the injury risk prediction.
        """)
        plot_feature_importance(model, df_processed.drop(['Injury_Risk', 'PLAYER_NAME'], axis=1).columns, top_n=15)
        
        st.subheader("SHAP Summary Plot")
        st.write("""
        SHAP values provide a more detailed view of feature importance. They show how each feature 
        impacts the model output for each prediction. 
        
        - Features are ranked by importance from top to bottom.
        - Colors indicate whether the feature value is high (red) or low (blue) for that observation.
        - The horizontal location shows whether the effect of that value caused a higher or lower prediction.
        """)
        explainer = shap.TreeExplainer(model)
        
        # Prepare the data for SHAP
        X = df_processed.drop(['Injury_Risk', 'PLAYER_NAME'], axis=1)
        X_sample = X.sample(min(100, len(X)))  # Sample up to 100 rows
        X_sample_scaled = scaler.transform(X_sample)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Ensure shap_values is a list with at least two elements
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_sample, feature_names=X_sample.columns, plot_type="bar", show=False)
        st.pyplot(fig)
    
    elif choice == "About":
        st.subheader("About")
        st.write("""
        This app predicts the injury risk of NBA players based on historical data. It uses a Random Forest Classifier 
        trained on player statistics and performance data.
        
        The model takes into account various factors such as player age, experience, position, and performance statistics 
        to estimate the likelihood of a player getting injured.
        
        For high-risk players, the app provides treatment recommendations from Auragens, a Mesenchymal Stem Cell clinic 
        located in Panama City, Panama. These recommendations are tailored based on the player's risk level.
        
        Key features of the app include:
        1. Injury risk prediction for individual players
        2. Player risk lookup for all players in the database
        3. Ability to add new player records
        4. Data visualizations to understand feature importance and model predictions
        
        Please note that while this tool can provide insights, it should not be used as the sole basis for medical or 
        strategic decisions. Always consult with medical professionals and team experts for comprehensive player evaluations.
        
        Data sources:
        - Player statistics: NBA API
        - Injury data: This model uses a simplified approach. In a real-world scenario, you would need to incorporate 
          actual injury reports and medical records, which are not publicly available due to privacy concerns.
        
        For more information on the data processing and model training, please refer to the `nba_data_processor.py` script.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("An error occurred in the Streamlit app")