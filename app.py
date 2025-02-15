# app.py
from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline, UserProfile
from src.exception import CustomException
import logging

app = Flask(__name__)
predictor = PredictionPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Course recommender system is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Generate course recommendations based on user profile"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400

        data = request.json
        required_fields = ['education', 'skills', 'career_goals', 'learning_objectives']
        
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields. Required: {required_fields}'
            }), 400

        # Create user profile
        profile = UserProfile(
            education=data['education'],
            skills=data['skills'],
            career_goals=data['career_goals'],
            learning_objectives=data['learning_objectives']
        )

        # Get number of recommendations (optional)
        top_n = int(data.get('top_n', 5))
        if top_n < 1 or top_n > 20:
            return jsonify({
                'status': 'error',
                'message': 'top_n must be between 1 and 20'
            }), 400

        # Generate recommendations
        result = predictor.predict(profile, top_n)
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(CustomException(e, sys))
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)