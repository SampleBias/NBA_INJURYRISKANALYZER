# data_processor.py

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, commonallplayers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = 'nba_data.pkl'
MODEL_PATH = 'nba_model.joblib'
SCALER_PATH = 'nba_scaler.joblib'
PROCESSED_DATA_PATH = 'nba_processed_data.pkl'

def load_and_process_data():
    logging.info("Starting data loading and processing")
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24"]
    
    # Use NBA API to get game data
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=seasons, league_id_nullable="00")
    games = gamefinder.get_data_frames()[0]
    
    # Get all players
    all_players = commonallplayers.CommonAllPlayers()
    players = all_players.get_data_frames()[0]
    
    logging.info(f"Games columns: {games.columns.tolist()}")
    logging.info(f"Players columns: {players.columns.tolist()}")
    
    # Create a dataset with unique teams and their most recent game stats
    df = games.sort_values('GAME_DATE').groupby('TEAM_ID').last().reset_index()
    
    # Now merge this with the players dataframe
    df = pd.merge(df, players, on='TEAM_ID', how='left')
    
    # Convert 'FROM_YEAR' and 'TO_YEAR' to numeric, replacing any non-numeric values with NaN
    df['FROM_YEAR'] = pd.to_numeric(df['FROM_YEAR'], errors='coerce')
    df['TO_YEAR'] = pd.to_numeric(df['TO_YEAR'], errors='coerce')
    
    # Add some basic calculations
    current_year = pd.to_datetime('today').year
    df['AGE'] = current_year - df['FROM_YEAR']
    df['SEASON_EXP'] = df['TO_YEAR'] - df['FROM_YEAR']
    
    # Remove rows with NaN values in key columns
    df = df.dropna(subset=['AGE', 'SEASON_EXP'])
    
    logging.info("Data loading and initial processing completed")
    return df

def preprocess_data(df):
    logging.info("Starting data preprocessing")
    features = [
        'DISPLAY_FIRST_LAST', 'AGE', 'SEASON_EXP',
        'PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT'
    ]
    available_features = [col for col in features if col in df.columns]
    df_processed = df[available_features].copy()
    
    # Create a simple injury risk indicator (this is a placeholder)
    df_processed['Injury_Risk'] = (df_processed['MIN'] < 10).astype(int)
    
    # We don't have 'POSITION' in our data, so we'll skip the one-hot encoding for now
    
    X = df_processed.drop(['Injury_Risk', 'DISPLAY_FIRST_LAST'], axis=1)
    y = df_processed['Injury_Risk']
    
    logging.info("Data preprocessing completed")
    return X, y, df_processed

def train_model(X, y):
    logging.info("Starting model training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    logging.info("Model training completed")
    return model, scaler

def main():
    # Load and process data
    df = load_and_process_data()
    df.to_pickle(DATA_PATH)
    logging.info(f"Raw data saved to {DATA_PATH}")
    
    # Preprocess data
    X, y, df_processed = preprocess_data(df)
    df_processed.to_pickle(PROCESSED_DATA_PATH)
    logging.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()
    