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

class InjuryRiskPredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.feature_names is not None:
            data = data.reindex(columns=self.feature_names, fill_value=0)
        return self.scaler.transform(data)

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.preprocess_data(data)
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.model.predict(X_scaled)
        return predictions, probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()

    def predict_single(self, input_data: dict) -> Tuple[int, float]:
        df = pd.DataFrame([input_data])
        predictions, probabilities = self.predict(df)
        return predictions[0], probabilities[0]

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
            return feature_imp.sort_values('importance', ascending=False).head(top_n)
        else:
            return pd.DataFrame()

    def plot_feature_importance(feature_imp: pd.DataFrame):
        if not feature_imp.empty:
        # Ensure 'importance' is numeric and 'feature' is string
          feature_imp['importance'] = pd.to_numeric(feature_imp['importance'], errors='coerce')
          feature_imp['feature'] = feature_imp['feature'].astype(str)
        
        # Remove any rows with NaN values
          feature_imp = feature_imp.dropna()
        
        # Sort the dataframe
          feature_imp = feature_imp.sort_values('importance', ascending=True)
        
        # Create the plot
          fig, ax = plt.subplots(figsize=(10, max(6, len(feature_imp) * 0.3)))
        
        # Plot horizontal bars
          bars = ax.barh(y=feature_imp['feature'], width=feature_imp['importance'])
        
        # Customize the plot
          ax.set_xlabel('Importance')
          ax.set_title('Feature Importance')
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
        
        # Add value labels to the end of each bar
          for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                    ha='left', va='center', fontweight='bold', fontsize=8)
        
        # Adjust layout and display the plot
          plt.tight_layout()
          st.pyplot(fig)
        else:
         st.write("Feature importance not available for this model.")

def get_auragens_recommendation(injury_risk_pred: int) -> str:
    if injury_risk_pred == 1:
        return ("Auragens Treatment Recommendation: "
                "IV infusion of 150 Million Mesenchymal Stem Cells (MSCs) systemically, "
                "combined with a local injection of 5-10 Million MSCs "
                "at the site of injury, administered over 3 consecutive days.")
    else:
        return ("Auragens Treatment Recommendation: "
                "Performance-enhancing injury prevention treatment consisting of "
                "intravenous (IV) infusion of 150 Million Mesenchymal Stem Cells (MSCs) "
                "administered over 3 consecutive days.")

def main():
    st.set_page_config(page_title="NBA Injury Risk Prediction App", page_icon="🏀")
    st.title("NBA Injury Risk Prediction App 🏀")
    
    df_processed = load_data()
    model, scaler = load_model()
    
    if df_processed is None or model is None or scaler is None:
        st.error("Required components are missing. App functionality will be limited.")
        return

    predictor = InjuryRiskPredictor(model, scaler)
    
    menu = ["Home", "Predict Injury Risk", "Player Risk Lookup", "Data Visualization", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to the NBA Injury Risk Prediction App")
        st.write("This app uses machine learning to predict the injury risk for NBA players based on their statistics and historical data.")
    
    elif choice == "Predict Injury Risk":
        st.subheader("Predict Injury Risk for a Player")
        
        with st.form("player_form"):
            player_name = st.text_input("Player Name")
            age = st.number_input("Age", min_value=18, max_value=50, value=25)
            season_exp = st.number_input("Years of Experience", min_value=0, max_value=30, value=3)
            pts = st.number_input("Points per Game", min_value=0.0, value=10.0)
            ast = st.number_input("Assists per Game", min_value=0.0, value=2.0)
            reb = st.number_input("Rebounds per Game", min_value=0.0, value=5.0)
            minutes_played = st.number_input("Minutes per Game", min_value=0.0, value=20.0)
            fg_pct = st.number_input("Field Goal Percentage", min_value=0.0, max_value=1.0, value=0.45)
            fg3_pct = st.number_input("3-Point Percentage", min_value=0.0, max_value=1.0, value=0.35)
            ft_pct = st.number_input("Free Throw Percentage", min_value=0.0, max_value=1.0, value=0.75)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'AGE': age, 'SEASON_EXP': season_exp,
                'PTS': pts, 'AST': ast, 'REB': reb, 'MIN': minutes_played,
                'FG_PCT': fg_pct, 'FG3_PCT': fg3_pct, 'FT_PCT': ft_pct
            }
            
            risk_prediction, risk_probability = predictor.predict_single(input_data)
            risk_label = 'High Risk' if risk_prediction == 1 else 'Low Risk'
            st.markdown(f"### Injury Risk Prediction for {player_name}: {risk_label}")
            st.markdown(f"#### Risk Probability: {risk_probability:.2f}")
            
            recommendation = get_auragens_recommendation(risk_prediction)
            st.markdown("---")
            st.markdown(f"### {recommendation}")
    
    elif choice == "Player Risk Lookup":
        st.subheader("NBA Players Injury Risk")
        
        df_for_prediction = df_processed.drop(['DISPLAY_FIRST_LAST', 'Injury_Risk'], axis=1, errors='ignore')
        predictions, probabilities = predictor.predict(df_for_prediction)
        df_results = pd.DataFrame({
            'DISPLAY_FIRST_LAST': df_processed['DISPLAY_FIRST_LAST'],
            'Injury_Risk_Prediction': predictions,
            'Injury_Risk_Probability': probabilities
        })
        
        player_names = df_processed['DISPLAY_FIRST_LAST'].unique()
        selected_player = st.selectbox("Select a Player", sorted(player_names))
        player_data = df_processed[df_processed['DISPLAY_FIRST_LAST'] == selected_player]
        player_result = df_results[df_results['DISPLAY_FIRST_LAST'] == selected_player].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<h2 style='font-size: 24px;'>{selected_player}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='font-size: 20px;'>Injury Risk Prediction: <strong>{'High' if player_result['Injury_Risk_Prediction'] == 1 else 'Low'} Risk</strong></h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='font-size: 20px;'>Risk Probability: <strong>{player_result['Injury_Risk_Probability']:.2f}</strong></h3>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Player Statistics")
            st.write(player_data[['AGE', 'SEASON_EXP', 'PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT']].T)
        
        recommendation = get_auragens_recommendation(player_result['Injury_Risk_Prediction'])
        st.markdown("---")
        st.markdown(f"### {recommendation}")

    elif choice == "Data Visualization":
        st.subheader("Data Visualization")
        
        st.write("This section provides insights into how our model makes predictions.")
        
        st.subheader("Feature Importance")
        feature_imp = predictor.get_feature_importance()
        if not feature_imp.empty:
            st.write("""
            Feature importance indicates how much each feature contributes to the model's predictions. 
            Higher values indicate greater importance in the model's decision-making process.
            """)
            plot_feature_importance(feature_imp)
        else:
            st.write("Feature importance is not available for this model.")
        
        st.subheader("SHAP Summary Plot")
        st.write("""
        SHAP (SHapley Additive exPlanations) values provide a detailed view of feature importance for each prediction. 
        They show how each feature impacts the model output across all predictions.
        """)
        
        if hasattr(predictor.model, 'predict_proba'):
            try:
                explainer = shap.TreeExplainer(predictor.model)
                X_sample = df_processed.drop(['DISPLAY_FIRST_LAST', 'Injury_Risk'], axis=1, errors='ignore').sample(n=min(100, len(df_processed)))
                X_sample_scaled = predictor.preprocess_data(X_sample)
                shap_values = explainer.shap_values(X_sample_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                                  X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                st.write(f"An error occurred while generating the SHAP plot: {str(e)}")
        else:
            st.write("SHAP values are not available for this model type.")
    
    elif choice == "About":
        st.subheader("About")
        st.write("This app predicts the injury risk of NBA players based on historical data using a Random Forest Classifier.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("An error occurred in the Streamlit app")
