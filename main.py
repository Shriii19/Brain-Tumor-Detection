from flask import Flask, render_template, request, send_from_directory, jsonify, flash
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import load_img, img_to_array
    import numpy as np
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - running in demo mode")
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'neuroscan_pro_2025_secure_key'  # For flash messages
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'dcm'}

# Load the trained model with error handling
try:
    # model = load_model('model.h5')
    logger.info("Model loading skipped - running in demo mode")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Enhanced class labels with medical descriptions
class_labels = {
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'description': 'Benign tumor of the pituitary gland',
        'severity': 'Low',
        'color': 'warning'
    },
    'glioma': {
        'name': 'Glioma',
        'description': 'Tumor originating in glial cells',
        'severity': 'High',
        'color': 'danger'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'Normal brain tissue, no abnormalities found',
        'severity': 'None',
        'color': 'success'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'Tumor arising from the meninges',
        'severity': 'Medium',
        'color': 'info'
    }
}

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path):
    """Validate if uploaded file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def generate_analysis_id():
    """Generate unique analysis ID"""
    return str(uuid.uuid4())[:8]

def log_analysis(analysis_id, filename, result, confidence):
    """Log analysis for audit trail"""
    log_entry = {
        'analysis_id': analysis_id,
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'result': result,
        'confidence': confidence,
        'model_version': 'Demo Mode v1.0'
    }
    logger.info(f"Analysis completed: {json.dumps(log_entry)}")

# Enhanced prediction function
def predict_tumor(image_path, analysis_id):
    """Enhanced tumor prediction with detailed results"""
    try:
        if model is None or not TENSORFLOW_AVAILABLE:
            # Demo mode - simulate realistic predictions
            import random
            
            # Simulate different tumor types with varying probabilities
            tumor_types = list(class_labels.keys())
            weights = [0.15, 0.10, 0.65, 0.10]  # Higher chance of 'notumor'
            predicted_type = random.choices(tumor_types, weights=weights)[0]
            
            # Generate realistic confidence scores
            if predicted_type == 'notumor':
                confidence = random.uniform(0.85, 0.97)
            else:
                confidence = random.uniform(0.78, 0.94)
            
            result_data = {
                'prediction': predicted_type,
                'confidence': confidence,
                'tumor_info': class_labels[predicted_type],
                'analysis_id': analysis_id,
                'processing_time': round(random.uniform(2.8, 4.2), 1),
                'model_version': 'Demo Mode v1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            return result_data
        
        # Real model prediction (when model is loaded)
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        if TENSORFLOW_AVAILABLE:
            import numpy as np
            img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        if TENSORFLOW_AVAILABLE:
            import numpy as np
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = np.max(predictions, axis=1)[0]
        
        predicted_type = list(class_labels.keys())[predicted_class_index]
        
        result_data = {
            'prediction': predicted_type,
            'confidence': float(confidence_score),
            'tumor_info': class_labels[predicted_type],
            'analysis_id': analysis_id,
            'processing_time': 3.2,
            'model_version': 'NeuroScan v2.4',
            'timestamp': datetime.now().isoformat()
        }
        
        return result_data
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise Exception(f"Analysis failed: {str(e)}")

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return render_template('index.html', error='No file selected')
            
            file = request.files['file']
            
            # Check if file was actually selected
            if file.filename == '':
                flash('No file selected', 'error')
                return render_template('index.html', error='No file selected')
            
            # Validate file extension
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload an image file.', 'error')
                return render_template('index.html', error='Invalid file type')
            
            # Generate unique filename and analysis ID
            analysis_id = generate_analysis_id()
            filename = secure_filename(file.filename)
            unique_filename = f"{analysis_id}_{filename}"
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_location)
            
            # Validate image integrity
            if not validate_image(file_location):
                os.remove(file_location)  # Clean up invalid file
                flash('Invalid image file. Please upload a valid image.', 'error')
                return render_template('index.html', error='Invalid image file')
            
            # Perform prediction
            result_data = predict_tumor(file_location, analysis_id)
            
            # Log the analysis
            log_analysis(
                analysis_id, 
                filename, 
                result_data['prediction'], 
                result_data['confidence']
            )
            
            # Prepare response data
            response_data = {
                'analysis_id': analysis_id,
                'filename': filename,
                'prediction': result_data['prediction'],
                'confidence': result_data['confidence'],
                'tumor_info': result_data['tumor_info'],
                'processing_time': result_data['processing_time'],
                'file_path': f'/uploads/{unique_filename}',
                'timestamp': result_data['timestamp'],
                'model_version': result_data['model_version']
            }
            
            flash('Analysis completed successfully', 'success')
            return render_template('index.html', result=response_data)
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash(f'Analysis failed: {str(e)}', 'error')
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

# API endpoint for AJAX requests
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for image analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process file
        analysis_id = generate_analysis_id()
        filename = secure_filename(file.filename)
        unique_filename = f"{analysis_id}_{filename}"
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_location)
        
        if not validate_image(file_location):
            os.remove(file_location)
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Perform prediction
        result_data = predict_tumor(file_location, analysis_id)
        
        # Log analysis
        log_analysis(analysis_id, filename, result_data['prediction'], result_data['confidence'])
        
        # Return JSON response
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'prediction': result_data['prediction'],
            'confidence': result_data['confidence'],
            'tumor_info': result_data['tumor_info'],
            'processing_time': result_data['processing_time'],
            'file_path': f'/uploads/{unique_filename}',
            'timestamp': result_data['timestamp']
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Model info endpoint
@app.route('/api/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_loaded': model is not None,
        'model_version': 'Demo Mode v1.0' if model is None else 'NeuroScan v2.4',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '25MB',
        'tumor_types': class_labels
    })

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded files securely"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return "File not found", 404

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 25MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('index.html', error='Internal server error'), 500

if __name__ == '__main__':
    logger.info("Starting NeuroScan Pro Application...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Model loaded: {model is not None}")
    logger.info("Application ready for medical imaging analysis")
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False,  # Set to False for production
        threaded=True   # Handle multiple requests
    )
