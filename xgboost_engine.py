
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostForecastEngine:
    def __init__(self, file_path, target_product='Castor'):
        self.file_path = file_path
        self.target_product = target_product
        self.reg = None
        self.data_series = None
        self.extended_data = None # Holds history + latest forecast
        self.feature_columns = None
        
    def load_and_train(self):
        """Loads data, trains XGBoost model, and prepares for forecasting."""
        logger.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise

        # Date column detection
        date_col = None
        for col in ['Expiry Date', 'Expiry_Date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
                break
        if not date_col:
             date_cols = [col for col in df.columns if 'date' in col.lower()]
             if date_cols:
                 date_col = date_cols[0]
        
        if not date_col:
            raise ValueError("Could not find date column in CSV.")

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # Filter Product
        if 'Product' in df.columns:
            df = df[df['Product'] == self.target_product].copy()

        price_col = 'Close' if 'Close' in df.columns else 'Price'
        
        # Aggregate daily
        data = df.groupby(date_col)[price_col].mean().to_frame()
        data = data.asfreq('D').fillna(method='ffill').fillna(method='bfill')
        
        # Determine "Today" cutoff (using system time)
        # Any data in the CSV after "today" should be ignored/truncated so the model
        # is forced to PREDICT it rather than read it as static history.
        cutoff_date = pd.Timestamp.now().normalize()
        
        if data.index.max() > cutoff_date:
            logger.info(f"Data extends into future ({data.index.max()}). Truncating to {cutoff_date}.")
            data = data[data.index <= cutoff_date]
            
        self.data_series = data
        self.price_col = price_col
        
        # --- Feature Engineering ---
        logger.info("Generating features...")
        data_features = self._create_features(data)
        
        # Lag Features
        LAGS = [1, 2, 3, 7, 14, 30, 60]
        for lag in LAGS:
            data_features[f'lag_{lag}'] = data_features[price_col].shift(lag)
            
        # Rolling Features
        WINDOWS = [7, 30, 60]
        for window in WINDOWS:
            data_features[f'rolling_mean_{window}'] = data_features[price_col].shift(1).rolling(window=window).mean()
            
        data_features = data_features.dropna()
        self.feature_columns = [c for c in data_features.columns if c != price_col]
        
        # --- Training ---
        logger.info("Training XGBoost model...")
        X = data_features[self.feature_columns]
        y = data_features[price_col]
        
        self.reg = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            objective='reg:squarederror'
        )
        self.reg.fit(X, y)
        logger.info("XGBoost training complete.")
        
        # Initialize extended data with history for forecasting context
        self.extended_data = data.copy()

    def _create_features(self, df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        return df

    def get_forecast(self, start_date, end_date):
        if self.reg is None:
            raise ValueError("Model not trained. Call load_and_train() first.")
            
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        last_history_date = self.data_series.index[-1]
        
        # Estimate volatility from recent history (std dev of last 30 days changes)
        recent_volatility = self.data_series[self.price_col].diff().tail(30).std()
        if pd.isna(recent_volatility): recent_volatility = 10.0 # Default fallback
        
        # Forecast needed from tomorrow until end_date
        forecast_dates_needed = pd.date_range(start=last_history_date + timedelta(days=1), end=end_dt, freq='D')
        
        current_data = self.data_series.copy()
        future_preds = {}
        
        if len(forecast_dates_needed) > 0:
            # We use a fixed seed for consistency across requests for the same date range if needed,
            # but for "simulation" feeling, random is fine. 
            # To avoid "jitter" between UI refreshes, maybe seed based on date? 
            # For now, let's keep it random to show movement or seed it.
            np.random.seed(42) 
            
            for date in forecast_dates_needed:
                # 1. Create feature row
                new_row = pd.DataFrame(index=[date])
                new_row = self._create_features(new_row)
                
                # 2. Calculate lags/rolling
                current_features = {}
                for col in self.feature_columns:
                    if col in new_row.columns:
                        current_features[col] = new_row[col].values[0]
                    elif 'lag_' in col:
                        lag = int(col.split('_')[1])
                        if len(current_data) >= lag:
                            current_features[col] = current_data[self.price_col].iloc[-lag]
                        else:
                            current_features[col] = np.nan
                    elif 'rolling_mean_' in col:
                        win = int(col.split('_')[2])
                        current_features[col] = current_data[self.price_col].iloc[-win:].mean()
                
                # 3. Predict
                X_feat = pd.DataFrame([current_features], index=[date])
                X_feat = X_feat[self.feature_columns]
                
                pred_price = float(self.reg.predict(X_feat)[0])
                
                # 4. Inject Noise (Market Volatility)
                # Add random fluctuation based on recent volatility
                noise = np.random.normal(0, recent_volatility * 0.8) 
                final_price = pred_price + noise
                
                future_preds[date] = final_price
                
                # 5. Append to history for next recursion
                new_data_row = pd.DataFrame({self.price_col: [final_price]}, index=[date])
                current_data = pd.concat([current_data, new_data_row])

        # Construct result
        final_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        result_list = []
        
        for d in final_dates:
            price = 0
            if d in future_preds:
                price = future_preds[d]
                is_forecast = True
            elif d in self.data_series.index:
                price = self.data_series.loc[d, self.price_col]
                is_forecast = False
            else:
                price = 0
                is_forecast = True
                
            result_list.append({
                "date": d.strftime("%Y-%m-%d"),
                "xgboost_price": round(price, 2),
                "is_forecast": is_forecast
            })
            
        return result_list
