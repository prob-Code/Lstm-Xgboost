
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for TensorFlow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not detected. LSTM model will be disabled.")
    TF_AVAILABLE = False

class ForecastEngine:
    def __init__(self, file_path, target_product='Castor'):
        self.file_path = file_path
        self.target_product = target_product
        self.look_back = 60
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.arima_fit = None
        self.lstm_model = None
        self.df_product = None
        self.data_series = None
        
    def load_data(self):
        logger.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise

        # Date column detection logic
        date_col = None
        possible_cols = ['Expiry Date', 'Expiry_Date', 'Date', 'DATE']
        for col in possible_cols:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
             # Fallback: search for any column containing 'date' or 'expiry'
             date_cols = [col for col in df.columns if 'date' in col.lower() or 'expiry' in col.lower()]
             if date_cols:
                 date_col = date_cols[0]
        
        if not date_col:
            raise ValueError("Could not find a recognizable Date column in the CSV file.")
            
        logger.info(f"Using date column: {date_col}")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Filter for target product
        if 'Product' in df.columns:
            df_filtered = df[df['Product'] == self.target_product].copy()
        else:
            df_filtered = df.copy() # Assume single product if no Product column
            
        df_filtered = df_filtered.sort_values(by=date_col)
        
        if df_filtered.empty:
            raise ValueError(f"No data found for product: {self.target_product}")

        # Group by date and mean close price
        price_col = 'Close' if 'Close' in df_filtered.columns else 'Price'
        if price_col not in df_filtered.columns:
             # Fallback to last column or specific logic
             price_col = df_filtered.columns[-1]

        data = df_filtered.groupby(date_col)[price_col].mean().to_frame()
        data.columns = ['Close'] # Standardize
        data = data.fillna(method='ffill')
        
        # Resample to daily frequency
        if not data.empty:
            full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
            data = data.reindex(full_date_range)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
        self.data_series = data.dropna()
        self.df_product = df_filtered
        logger.info(f"Data loaded successfully. Total points: {len(self.data_series)}")

    def train(self):
        if self.data_series is None or self.data_series.empty:
            raise ValueError("No data to train on. Call load_data() first.")
            
        train_size = int(len(self.data_series) * 0.8)
        # Ensure we have enough data
        if train_size < 10:
             logger.warning("Not enough data to split for training. Training on full dataset.")
             train_size = len(self.data_series)
             
        train_data = self.data_series[:train_size]
        
        # --- ARIMA Training ---
        logger.info("Training ARIMA model...")
        try:
            # Simple ARIMA order (5,1,0)
            arima_model = ARIMA(train_data['Close'], order=(5, 1, 0), freq='D')
            self.arima_fit = arima_model.fit()
            logger.info("ARIMA model trained successfully.")
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            self.arima_fit = None

        # --- LSTM Training ---
        if TF_AVAILABLE and len(self.data_series) > self.look_back + 10:
            logger.info("Training LSTM model...")
            try:
                # Clear session to avoid leaks
                K.clear_session()
                
                scaled_data = self.scaler.fit_transform(self.data_series['Close'].values.reshape(-1, 1))
                train_scaled = scaled_data[:train_size]
                
                X_train, y_train = self._create_sequences(train_scaled, self.look_back)
                
                if len(X_train) > 0:
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dense(units=1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    # Train with few epochs for speed in this demo
                    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                    self.lstm_model = model
                    logger.info("LSTM model trained successfully.")
                else:
                    logger.warning("Not enough data to create sequences for LSTM.")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                self.lstm_model = None
        else:
            logger.warning("Skipping LSTM: TensorFlow not available or insufficient data.")

    def _create_sequences(self, data, look_back):
        X, Y = [], []
        if len(data) <= look_back:
            return np.array(X), np.array(Y)
            
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    def get_forecast(self, start_date, end_date):
        """
        Generates forecast for the specified date range.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if start_dt > end_dt:
            raise ValueError("StartDate cannot be after EndDate")

        forecast_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        result_dates = [d.strftime("%Y-%m-%d") for d in forecast_dates]
        arima_vals = [0] * len(forecast_dates)
        lstm_vals = [0] * len(forecast_dates)
        
        # --- Generate ARIMA Forecast ---
        if self.arima_fit:
            try:
                # Predict gives values for the dates.
                # Note: predict() handles date indices if model was trained with them.
                # However, predicting far into future might require specific handling.
                # We use simple prediction wrapper.
                pred_res = self.arima_fit.predict(start=forecast_dates[0], end=forecast_dates[-1], dynamic=False)
                arima_vals = [round(x, 2) for x in pred_res.values]
            except Exception as e:
                logger.error(f"ARIMA prediction error: {e}")
                # Fallbck: Constant last value or error
                pass

        # --- Generate LSTM Forecast ---
        if self.lstm_model and self.data_series is not None:
            try:
                # We need to project into the future from the last known data point
                last_known_date = self.data_series.index[-1]
                
                # Check if requested range is in future or past
                # For this implementation, we will ONLY forecast the future part relative to training data.
                # For past dates, we ideally return actuals or training predictions.
                
                # Prepare sequence from the very end of data
                full_scaled = self.scaler.transform(self.data_series['Close'].values.reshape(-1, 1))
                curr_sequence = full_scaled[-self.look_back:].copy()
                
                # We start predicting from last_known_date + 1 day
                days_until_end = (end_dt - last_known_date).days
                
                if days_until_end > 0:
                    future_preds = []
                    # Predict iteratively
                    temp_seq = curr_sequence.reshape(1, self.look_back, 1)
                    
                    for _ in range(days_until_end):
                        next_val_scaled = self.lstm_model.predict(temp_seq, verbose=0)
                        future_preds.append(next_val_scaled[0, 0])
                        
                        # Update sequence: shift left, add new value
                        # curr_sequence shape: (look_back, 1) -> (1, look_back, 1)
                        # We need to act on inner array
                        new_row = next_val_scaled # shape (1,1)
                        
                        # Extract 1D array from temp_seq to shift
                        # temp_seq is (1, 60, 1)
                        seq_inner = temp_seq[0] # (60, 1)
                        seq_shifted = np.append(seq_inner[1:], new_row, axis=0)
                        temp_seq = seq_shifted.reshape(1, self.look_back, 1)

                    # Inverse transform
                    future_prices = self.scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
                    
                    # Map to dates
                    future_date_range = pd.date_range(start=last_known_date + timedelta(days=1), periods=days_until_end, freq='D')
                    future_series = pd.Series(future_prices, index=future_date_range)
                    
                    # Fill requested result list
                    for i, d in enumerate(forecast_dates):
                        if d in future_series.index:
                            lstm_vals[i] = round(future_series[d], 2)
                        elif d <= last_known_date:
                            # If date is in past, return actual if available?
                            if d in self.data_series.index:
                                lstm_vals[i] = round(self.data_series.loc[d, 'Close'], 2)
                
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")
        
        # Calculate Average
        avg_vals = []
        for a, l in zip(arima_vals, lstm_vals):
            valid_vals = []
            if a != 0: valid_vals.append(a)
            if l != 0: valid_vals.append(l)
            
            if valid_vals:
                avg_vals.append(round(sum(valid_vals)/len(valid_vals), 2))
            else:
                avg_vals.append(0)

        return {
            "dates": result_dates,
            "arima": arima_vals,
            "lstm": lstm_vals,
            "average": avg_vals
        }
