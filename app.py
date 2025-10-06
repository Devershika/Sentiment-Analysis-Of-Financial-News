from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model
model = None
vectorizer = None
label_encoder = None

def load_models():
    """Load all model artifacts"""
    global model, vectorizer, label_encoder
    
    try:
        logger.info("Loading models...")
        
        # Load trained model
        with open('src/components/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ Model loaded")
        
        # Load vectorizer
        with open('src/components/best_model.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("✓ Vectorizer loaded")
        
        # Load label encoder
        with open('data\label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("✓ Label encoder loaded")
        
        logger.info("=" * 50)
        logger.info("All models loaded successfully!")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Make sure these files exist:")
        logger.error("  - trained_models/best_model.pkl")
        logger.error("  - trained_models/vectorizer.pkl")
        logger.error("  - preprocessed_data/label_encoder.pkl")
        return False

def predict_sentiment(text):
    """
    Predict sentiment for given text
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Vectorize input
        X = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            confidence_scores = {
                label: float(probabilities[i])
                for i, label in enumerate(label_encoder.classes_)
            }
            confidence = float(max(probabilities))
        else:
            confidence_scores = {predicted_label: 1.0}
            confidence = 1.0
        
        return {
            'text': text,
            'sentiment': predicted_label,
            'confidence': confidence,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# ============================================
# API ROUTES
# ============================================

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Financial Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Analyze sentiment',
            '/batch-predict': 'POST - Analyze multiple texts'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is None or vectorizer is None or label_encoder is None:
        return jsonify({
            'status': 'error',
            'message': 'Models not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'API is ready'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single text
    
    Expected JSON:
    {
        "text": "Your financial news text here"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        
        # Validation
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        # Make prediction
        result = predict_sentiment(text)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict sentiment for multiple texts
    
    Expected JSON:
    {
        "texts": ["text1", "text2", "text3"]
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        texts = data.get('texts', [])
        
        # Validation
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts format. Expected array of strings.'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts allowed per request'}), 400
        
        # Filter valid texts
        valid_texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 10]
        
        if not valid_texts:
            return jsonify({'error': 'No valid texts provided (minimum 10 characters each)'}), 400
        
        # Make predictions
        results = [predict_sentiment(text) for text in valid_texts]
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in /batch-predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/predict', '/batch-predict']
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Financial Sentiment Analysis API")
    print("=" * 60)
    
    # Load models
    if not load_models():
        print("\nFailed to load models. Exiting...")
        exit(1)
    
    print("\nModels loaded successfully!")
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nTest it with:")
    print("   curl http://localhost:5000/health")
    print("\n" + "=" * 60 + "\n")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Set to False in production
    )