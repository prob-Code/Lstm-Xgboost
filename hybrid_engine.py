
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from xgboost_engine import XGBoostForecastEngine
from model_engine import ForecastEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridForecastEngine:
    def __init__(self, file_path, target_product='Castor'):
        self.file_path = file_path
        self.target_product = target_product
        
        # Initialize sub-engines
        self.xgboost_engine = XGBoostForecastEngine(file_path, target_product)
        self.ml_engine = ForecastEngine(file_path, target_product)
        
        # Default Weights
        self.weights = {
            'xgboost': 0.4,
            'lstm': 0.4,
            'arima': 0.2
        }

    def load_and_train(self):
        """Initializes and trains all underlying models."""
        logger.info("Initializing Hybrid Engine components...")
        
        # Train XGBoost
        logger.info("Training XGBoost component...")
        self.xgboost_engine.load_and_train()
        
        # Train ML Engine (ARIMA + LSTM)
        logger.info("Training ARIMA/LSTM components...")
        self.ml_engine.load_data()
        self.ml_engine.train()
        
        logger.info("Hybrid Engine training complete.")

    def get_forecast(self, start_date, end_date):
        """
        Generates a hybrid forecast by ensembling XGBoost, LSTM, and ARIMA.
        """
        logger.info(f"Generating Hybrid forecast from {start_date} to {end_date}...")
        
        # 1. Get XGBoost Forecast
        xgb_results = self.xgboost_engine.get_forecast(start_date, end_date)
        # Convert list of dicts to a dict keyed by date for easy lookup
        xgb_lookup = {item['date']: item['xgboost_price'] for item in xgb_results}
        
        # 2. Get ARIMA/LSTM Forecast
        ml_results = self.ml_engine.get_forecast(start_date, end_date)
        # ml_results structure: {"dates": [...], "arima": [...], "lstm": [...], "average": [...]}
        
        hybrid_forecast = []
        
        for i, date_str in enumerate(ml_results['dates']):
            arima_val = ml_results['arima'][i]
            lstm_val = ml_results['lstm'][i]
            xgb_val = xgb_lookup.get(date_str, 0)
            
            # Weighted ensemble
            # Only include models that returned non-zero values
            available_models = []
            weighted_sum = 0
            total_weight = 0
            
            if arima_val > 0:
                weighted_sum += arima_val * self.weights['arima']
                total_weight += self.weights['arima']
                available_models.append('arima')
            
            if lstm_val > 0:
                weighted_sum += lstm_val * self.weights['lstm']
                total_weight += self.weights['lstm']
                available_models.append('lstm')
                
            if xgb_val > 0:
                weighted_sum += xgb_val * self.weights['xgboost']
                total_weight += self.weights['xgboost']
                available_models.append('xgboost')
            
            # Re-normalize weights if some models are missing
            if total_weight > 0:
                hybrid_price = round(weighted_sum / total_weight, 2)
            else:
                hybrid_price = 0
                
            hybrid_forecast.append({
                "date": date_str,
                "hybrid_price": hybrid_price,
                "xgboost_price": xgb_val,
                "arima_price": arima_val,
                "lstm_price": lstm_val,
                "models_used": available_models
            })
            
        return hybrid_forecast

if __name__ == "__main__":
    # Test the hybrid engine
    DATA_FILE = 'daily_oilseeds_full_ml_dataset_2015_01_01_2025_12_02.csv'
    engine = HybridForecastEngine(DATA_FILE)
    engine.load_and_train()
    
    forecast = engine.get_forecast("2026-01-01", "2026-01-10")
    print("\n--- Hybrid Forecast Sample ---")
    for day in forecast[:5]:
        print(day)
