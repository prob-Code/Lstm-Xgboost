import os
import logging
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid_engine import HybridForecastEngine

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridAPI")

# Configuration
DATA_FILE = 'daily_oilseeds_full_ml_dataset_2015_01_01_2025_12_02.csv'
API_KEYS_FILE = 'api_keys.json'
PRODUCT_NAME = 'Castor'

# ============= API KEY MANAGEMENT =============
def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_api_keys(keys):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def verify_api_key(key):
    api_keys = load_api_keys()
    if key in api_keys and api_keys[key].get('active', False):
        api_keys[key]['last_used'] = datetime.now().isoformat()
        api_keys[key]['requests_count'] = api_keys[key].get('requests_count', 0) + 1
        save_api_keys(api_keys)
        return True
    return False

# Global engine instance
hybrid_engine = None

def get_engine():
    global hybrid_engine
    if hybrid_engine is None:
        logger.info("Initializing Hybrid Engine for the first time...")
        hybrid_engine = HybridForecastEngine(DATA_FILE, PRODUCT_NAME)
        hybrid_engine.load_and_train()
    return hybrid_engine

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Hybrid Price Forecasting API",
        "engine_initialized": hybrid_engine is not None
    })

@app.route('/api/forecast', methods=['GET', 'POST'])
def forecast():
    # Verify API key
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if not api_key or not verify_api_key(api_key):
        return jsonify({
            'status': 'error',
            'message': 'Invalid or missing API key.'
        }), 401
        
    try:
        # Get parameters
        if request.method == 'POST':
            data = request.json or {}
        else:
            data = request.args.to_dict()
            
        start_date = data.get('start_date', '2026-01-01')
        end_date = data.get('end_date', '2026-01-15')
        product = data.get('product', PRODUCT_NAME)
        
        # Ensure engine is ready
        engine = get_engine()
        
        # Override weights if provided
        if 'weights' in data:
            try:
                # Expecting something like {"xgboost": 0.5, "lstm": 0.3, "arima": 0.2}
                new_weights = data['weights']
                if sum(new_weights.values()) > 0:
                    engine.weights.update(new_weights)
                    logger.info(f"Using custom weights: {engine.weights}")
            except Exception as e:
                logger.warning(f"Failed to parse custom weights: {e}")

        # Generate Hybrid Forecast
        results = engine.get_forecast(start_date, end_date)
        
        # Calculate summary stats
        prices = [r['hybrid_price'] for r in results if r['hybrid_price'] > 0]
        avg_price = round(sum(prices) / len(prices), 2) if prices else 0

        # Add compatibility field
        for r in results:
            r['average_price'] = r['hybrid_price']
        
        return jsonify({
            "status": "success",
            "product": product,
            "forecast": results,
            "summary": {
                "avg_price": avg_price,
                "min_price": min(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "days": len(results)
            },
            "weights": engine.weights,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize engine on startup
    logger.info("Pre-training models...")
    get_engine()
    
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Hybrid API starting on port {port}")
    app.run(debug=False, port=port, host='0.0.0.0')
