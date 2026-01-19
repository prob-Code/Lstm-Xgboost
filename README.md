# 📈 Hybrid Price Forecasting: XGBoost + LSTM + ARIMA

A high-performance, production-ready price forecasting system that ensembles three powerful models to predict commodity prices (specifically Castor oilseeds) with high accuracy and stability.

## 🚀 Key Features

- **Hybrid Ensemble Logic**: Combines three distinct modeling approaches:
  - **XGBoost**: Captures complex non-linear relationships using lag features and rolling statistics.
  - **LSTM (Long Short-Term Memory)**: A deep learning approach that understands long-term temporal dependencies.
  - **ARIMA**: A statistical model that provides a solid baseline and handles trend stability.
- **Weighted Averaging**: The system uses a weighted ensemble (Default: 40% XGBoost, 40% LSTM, 20% ARIMA) to produce a single optimized forecast.
- **Interactive Dashboard**: A modern, real-time UI built with Plotly for visualizing individual and ensemble forecasts.
- **Production API**: A Flask-based API with API Key authentication, summary statistics, and customizable forecast ranges.

## 📂 Project Structure

- `hybrid_api.py`: The main entry point. Flask server providing the API endpoints.
- `hybrid_engine.py`: The ensembling logic that synchronizes predictions from all three models.
- `xgboost_engine.py`: Implementation of the XGBoost regressor with recursive forecasting logic.
- `model_engine.py`: Implementation of the LSTM (TensorFlow/Keras) and ARIMA models.
- `forecast_dashboard_v2.html`: Modern dashboard for visualization and data interaction.
- `daily_oilseeds_...csv`: Optimized historical dataset for training.

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the API
Start the Hybrid forecasting server:

```bash
python hybrid_api.py
```
*The API will start on `http://127.0.0.1:7860` by default.*

### 3. Open the Dashboard
Simply open `forecast_dashboard_v2.html` in any browser to see the live charts and ensemble predictions.

## 🔌 API Documentation

### Get Hybrid Forecast
**Endpoint:** `POST /api/forecast`  
**Headers:** `X-API-Key: castor_d167aa169b5e4219a66779e45fbaaefe`

**Payload:**
```json
{
  "start_date": "2026-01-01",
  "end_date": "2026-01-15",
  "weights": {
    "xgboost": 0.5,
    "lstm": 0.3,
    "arima": 0.2
  }
}
```

## 🧠 How the Hybrid Engine Works

1. **XGBoost**: We generate features like `dayofweek`, `rolling_mean`, and `lags` (1, 7, 30 days). The model predicts recursively into the future.
2. **LSTM**: We scale the data using `MinMaxScaler` and create sequences of 60 days to train the neural network.
3. **ARIMA**: We fit a (5,1,0) order model to capture the autoregressive components of the series.
4. **Ensemble**: The engine calculates a weighted sum of the three outputs. If one model fails or is missing, it dynamically re-normalizes the weights across the remaining models.

## 🔐 Security
API keys are managed via `api_keys.json`. You can revoke or add keys to control access to the forecasting endpoints.

---
Created for **AgentCrafter** | Optimized for deployment on Hugging Face and GitHub.
