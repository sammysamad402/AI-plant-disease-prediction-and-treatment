from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import os, time, io, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2
from typing import List, Dict, Any, Optional
import logging
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def clean_numpy_types(obj):
    """
    Recursively converts NumPy scalars (bool_, float64, int64, etc.) to native Python types.
    Handles dicts, lists, and primitives. Leaves other types unchanged.
    """
    if isinstance(obj, dict):
        return {key: clean_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_numpy_types(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:  # Scalar array
        return clean_numpy_types(obj.item())
    else:
        return obj


app = FastAPI(title="Advanced Plant Disease Detection System")

# Configuration
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# MongoDB setup
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.get_database("plant_disease_db")
collection = db.get_collection("detections")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease classes with detailed information
CLASS_NAMES = ['anthracnose', 'healthy', 'leaf_crinkle', 'powdery_mildew', 'yellow_mosaic']

DISEASE_INFO = {
    'anthracnose': {
        'severity': 'High',
        'color': '#FF4444',
        'description': 'Fungal disease causing dark lesions on leaves and stems',
        'symptoms': ['Dark, sunken spots on leaves', 'Brown or black lesions', 'Leaf yellowing and drop'],
        'treatment': 'Apply copper-based fungicides, remove infected plant parts, ensure good air circulation',
        'prevention': 'Avoid overhead watering, plant resistant varieties, maintain proper spacing',
        'organic_solutions': ['Neem oil spray', 'Baking soda solution', 'Compost tea application']
    },
    'healthy': {
        'severity': 'None',
        'color': '#44FF44',
        'description': 'Plant appears healthy with no visible disease symptoms',
        'symptoms': ['Green, vibrant foliage', 'No discoloration or lesions', 'Normal growth patterns'],
        'treatment': 'No treatment needed - maintain current care practices',
        'prevention': 'Continue regular monitoring, maintain optimal growing conditions',
        'organic_solutions': ['Regular inspection', 'Balanced nutrition', 'Proper watering schedule']
    },
    'leaf_crinkle': {
        'severity': 'Medium',
        'color': '#FF8844',
        'description': 'Viral disease causing leaf distortion and reduced photosynthesis',
        'symptoms': ['Wrinkled or puckered leaves', 'Stunted growth', 'Yellowing between veins'],
        'treatment': 'Remove affected leaves immediately, control insect vectors, quarantine infected plants',
        'prevention': 'Use virus-free planting material, control aphids and whiteflies, maintain plant hygiene',
        'organic_solutions': ['Reflective mulch', 'Beneficial insect habitat', 'Quarantine new plants']
    },
    'powdery_mildew': {
        'severity': 'Medium',
        'color': '#FFAA44',
        'description': 'Fungal disease creating white powdery coating on leaf surfaces',
        'symptoms': ['White powdery spots on leaves', 'Leaf curling and distortion', 'Reduced plant vigor'],
        'treatment': 'Apply sulfur-based fungicides, improve air circulation, reduce humidity around plants',
        'prevention': 'Avoid overhead watering, ensure proper plant spacing, choose resistant varieties',
        'organic_solutions': ['Milk spray (1:10 ratio)', 'Baking soda solution', 'Horticultural oils']
    },
    'yellow_mosaic': {
        'severity': 'High',
        'color': '#FFFF44',
        'description': 'Viral disease causing yellow mottling and severe yield reduction',
        'symptoms': ['Yellow patches on leaves', 'Mosaic-like patterns', 'Stunted plant growth'],
        'treatment': 'Remove infected plants immediately, control whitefly vectors, use resistant varieties',
        'prevention': 'Monitor for whiteflies, use yellow sticky traps, plant virus-resistant cultivars',
        'organic_solutions': ['Vector control with beneficial insects', 'Reflective mulches', 'Crop rotation']
    }
}

# Model loading
# Model loading
MODEL_PATH = os.environ.get("MODEL_PATH", "BPLD_CNN_model.h5")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully: {MODEL_PATH}")
        
        # ADD MODEL DEBUG INFO
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info(f"Number of classes: {model.output_shape[1]}")
        
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    logger.error(f'Model load error: {e}')

# Temporary test function - add this AFTER model loading
def test_model_predictions():
    """Test if model can predict different classes"""
    if model is not None:
        try:
            logger.info("=== RUNNING MODEL PREDICTION TEST ===")
            # Test with random images to see class distribution
            for i in range(3):  # Test 3 times
                test_img = np.random.random((1, 224, 224, 3)).astype('float32')
                test_pred = model.predict(test_img, verbose=0)
                predicted_idx = np.argmax(test_pred[0])
                confidence = test_pred[0][predicted_idx]
                predicted_class = CLASS_NAMES[predicted_idx]
                
                logger.info(f"Test {i+1}: Predicted {predicted_class} with {confidence:.3f} confidence")
                logger.info(f"All scores: {dict(zip(CLASS_NAMES, test_pred[0]))}")
            
            logger.info("=== MODEL TEST COMPLETE ===")
        except Exception as e:
            logger.error(f"Model test failed: {e}")

# Run the test when server starts - THIS LINE WAS MISSING!
test_model_predictions()
class WeatherEnhancer:
    """Handles image enhancement for various weather conditions"""
    
    @staticmethod
    def detect_weather_conditions(image_array: np.ndarray) -> Dict[str, Any]:
        """Detect weather conditions from image characteristics"""
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image_array * 255).astype(np.uint8)
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # Detect conditions
        is_foggy = mean_intensity > 180 and contrast < 0.3
        is_dark = mean_intensity < 80
        is_low_contrast = contrast < 0.2
        
        return {
    'is_foggy': bool(is_foggy),
    'is_dark': bool(is_dark),
    'is_low_contrast': bool(is_low_contrast),
    'mean_intensity': float(mean_intensity),
    'contrast': float(contrast)
}

    
    @staticmethod
    def enhance_for_weather(image_array: np.ndarray, weather_conditions: Dict) -> np.ndarray:
        """Apply weather-specific enhancements"""
        enhanced = image_array.copy()
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray((enhanced * 255).astype(np.uint8))
        
        # Apply enhancements based on conditions
        if weather_conditions['is_foggy']:
            # Increase contrast and reduce haze
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.2)
        
        if weather_conditions['is_dark']:
            # Brighten dark images
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.3)
        
        if weather_conditions['is_low_contrast']:
            # Increase contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.4)
        
        # Convert back to numpy array
        enhanced = np.array(pil_image).astype('float32') / 255.0
        return enhanced

class AdversarialDefense:
    """Advanced adversarial attack detection and defense"""
    
    @staticmethod
    def detect_adversarial_patterns(image_array: np.ndarray) -> Dict[str, Any]:
        """Detect potential adversarial attacks"""
        # Calculate gradient magnitude
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image_array * 255).astype(np.uint8)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Statistical analysis
        high_freq_energy = np.mean(gradient_magnitude)
        gradient_variance = np.var(gradient_magnitude)
        
        # Noise detection
        noise_threshold = 15
        is_adversarial = high_freq_energy > noise_threshold and gradient_variance > 100
        
        return {
            'is_adversarial': bool(is_adversarial),
            'confidence': float(min(high_freq_energy / noise_threshold, 1.0)),
            'high_freq_energy': float(high_freq_energy),
            'gradient_variance': float(gradient_variance)
        }
    
    @staticmethod
    def apply_defense_mechanisms(image_array: np.ndarray) -> np.ndarray:
        """Apply multiple defense mechanisms"""
        defended = image_array.copy()
        
        # Apply Gaussian blur to reduce high-frequency noise
        defended = cv2.GaussianBlur(defended, (3, 3), 0.8)
        
        # Apply median filter for impulse noise
        defended_uint8 = (defended * 255).astype(np.uint8)
        defended_uint8 = cv2.medianBlur(defended_uint8, 3)
        defended = defended_uint8.astype('float32') / 255.0
        
        return defended


class PlantDetector:
    """Detects if image contains a plant before disease classification"""
    
    @staticmethod
    def is_plant_image(image_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect if image contains plant/leaf based on color and texture analysis
        """
        # Convert to HSV for better color analysis
        image_uint8 = (image_array * 255).astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        
        # Define green color range (plants are typically green)
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        lower_green2 = np.array([25, 30, 30])
        upper_green2 = np.array([95, 255, 255])
        
        # Create masks for green colors
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage of green pixels
        green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100
        
        # Texture analysis
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        
        # Calculate color variation
        std_r = np.std(image_array[:, :, 0])
        std_g = np.std(image_array[:, :, 1])
        std_b = np.std(image_array[:, :, 2])
        color_variation = (std_r + std_g + std_b) / 3
        
        # Decision logic
        is_plant = False
        confidence = 0.0
        reason = "Not a plant"
        
        if green_percentage > 15:
            if edge_density > 5 and edge_density < 40:
                if color_variation > 0.05:
                    is_plant = True
                    confidence = min((green_percentage / 50) * 100, 100)
                    reason = "Plant detected based on green content and texture"
        
        # Skin tone detection (reject human faces/hands)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = (np.sum(skin_mask > 0) / skin_mask.size) * 100
        
        if skin_percentage > 30:
            is_plant = False
            confidence = 0.0
            reason = "Human skin detected - not a plant"
        
        return {
            'is_plant': bool(is_plant),
            'confidence': float(confidence),
            'green_percentage': float(green_percentage),
            'edge_density': float(edge_density),
            'skin_detected': bool(skin_percentage > 30),
            'reason': reason
        }
    # REMOVE the apply_defense_mechanisms method from PlantDetector - it doesn't belong here

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> tuple:
    """Advanced image preprocessing with plant detection, weather and adversarial handling"""
    try:
        logger.info(f"Starting preprocessing for: {image_path}")
        
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        logger.info("Image loaded successfully")
        
        img_array = np.array(img.resize(target_size)).astype('float32') / 255.0
        logger.info(f"Image resized to {target_size}, array shape: {img_array.shape}")
        
        # STEP 1: Check if image contains a plant
        plant_detector = PlantDetector()
        plant_check = plant_detector.is_plant_image(img_array)
        logger.info(f"Plant detection completed: {plant_check}")
        
        if not plant_check['is_plant']:
            logger.info(f"No plant detected: {plant_check['reason']}")
            return None, None, {'plant_detected': False, 'reason': plant_check['reason']}
        
        logger.info("Plant detected, proceeding with weather analysis")
        
        # STEP 2: Weather condition analysis
        weather_enhancer = WeatherEnhancer()
        weather_conditions = weather_enhancer.detect_weather_conditions(img_array)
        logger.info(f"Weather analysis completed: {weather_conditions}")
        
        # Apply weather enhancements
        if any(weather_conditions[key] for key in ['is_foggy', 'is_dark', 'is_low_contrast']):
            img_array = weather_enhancer.enhance_for_weather(img_array, weather_conditions)
            logger.info(f"Applied weather enhancements: {weather_conditions}")
        
        # STEP 3: Adversarial detection and defense
        defense_system = AdversarialDefense()
        adv_analysis = defense_system.detect_adversarial_patterns(img_array)
        logger.info(f"Adversarial analysis completed: {adv_analysis}")
        
        if adv_analysis['is_adversarial']:
            # FIXED: Changed from apply_defense_mechanism to apply_defense_mechanisms
            img_array = defense_system.apply_defense_mechanisms(img_array)
            logger.warning(f"Adversarial attack detected and defended: confidence {adv_analysis['confidence']:.3f}")
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        logger.info(f"Final batch shape: {img_batch.shape}, range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        
        
        # Combine all analysis
        full_analysis = {
            **plant_check,
            **adv_analysis, 
            **weather_conditions,
            'plant_detected': True
        }
        
        logger.info("Preprocessing completed successfully")
        return img_batch, img_array, full_analysis
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}", exc_info=True)
        return None, None, None
@app.post('/detect')
async def detect_disease(file: UploadFile = File(...), device_id: str = Form(None)):
    """Comprehensive disease detection with weather handling"""
    try:
        if model is None:
            return JSONResponse({'status': 'error', 'detail': 'Model not loaded'}, status_code=500)
        
        # Save uploaded file
        contents = await file.read()
        timestamp = int(time.time())
        device = device_id or "device"
        filename = f"{device}_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Advanced preprocessing with plant detection
                # Advanced preprocessing with plant detection
        processed_img, enhanced_img, analysis = preprocess_image(filepath)
        
        # DEBUG: Check what the model actually receives
        if processed_img is not None:
            logger.info(f"Image stats - Min: {processed_img.min():.3f}, Max: {processed_img.max():.3f}, Mean: {processed_img.mean():.3f}")
            # Save a debug image to see what the model sees
            debug_img = (processed_img[0] * 255).astype(np.uint8)
            debug_filename = f"debug_model_input_{timestamp}.jpg"
            debug_filepath = os.path.join(UPLOAD_DIR, debug_filename)
            Image.fromarray(debug_img).save(debug_filepath)
            logger.info(f"Saved model input as: {debug_filename}")
        
        # Check if no plant was detected
        if analysis and not analysis.get('plant_detected', False):
            return JSONResponse({
                'status': 'error',
                'detail': analysis.get('reason', 'No plant detected in image'),
                'message': 'Please upload an image containing plant leaves for disease detection',
                'plant_detected': False
            }, status_code=400)
        
        if processed_img is None:
            return JSONResponse({'status': 'error', 'detail': 'Failed to process image'}, status_code=400)
        # Clean analysis to remove NumPy types early
                # Clean analysis to remove NumPy types early
        analysis = clean_numpy_types(analysis or {})
        
        # TEMPORARY FIX: Try common preprocessing methods
                # TEMPORARY FIX: Try common preprocessing methods
        if processed_img is not None:
            # Method 1: ImageNet normalization 
            imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
            imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
            processed_img = (processed_img - imagenet_mean) / imagenet_std
            logger.info("Applied ImageNet normalization with proper reshaping")
        # Make prediction - ADD DEBUGGING
               # Make prediction - ADD DEBUGGING
        predictions = model.predict(processed_img, verbose=0)
        
        # DEBUG: Print all prediction scores
        logger.info(f"Raw predictions: {predictions[0]}")
        
        # Convert numpy floats to Python floats for logging
        prediction_scores = {class_name: float(score) for class_name, score in zip(CLASS_NAMES, predictions[0])}
        logger.info(f"Prediction scores: {prediction_scores}")
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        logger.info(f"Predicted: {predicted_class} (index: {predicted_class_idx}) with confidence: {confidence:.4f}")
        
        # ADD CONFIDENCE THRESHOLD
        CONFIDENCE_THRESHOLD = 0.6  # Require 60% confidence
        
        if confidence < CONFIDENCE_THRESHOLD:
            # If low confidence, default to "needs expert review" or "healthy"
            original_prediction = predicted_class
            predicted_class = 'healthy'
            confidence = 0.5  # Medium confidence for uncertain cases
            logger.warning(f"Low confidence prediction for {original_prediction} ({confidence:.3f}), defaulting to 'healthy'")
        # Get detailed disease information
        disease_data = DISEASE_INFO.get(predicted_class, {})
        
        # Create comprehensive prediction result
        prediction_result = {
            'disease_name': predicted_class.replace('_', ' ').title(),
            'confidence': confidence,
            'severity': disease_data.get('severity', 'Unknown'),
            'description': disease_data.get('description', 'No description available'),
            'symptoms': disease_data.get('symptoms', []),
            'treatment': disease_data.get('treatment', 'Consult agricultural expert'),
            'prevention': disease_data.get('prevention', 'Follow general plant care practices'),
            'organic_solutions': disease_data.get('organic_solutions', []),
            'plant_confidence': analysis.get('confidence', 0),
            'weather_conditions': {
                'foggy_detected': bool(analysis.get('is_foggy', False)),
                'dark_conditions': bool(analysis.get('is_dark', False)),
                'low_contrast': bool(analysis.get('is_low_contrast', False))
            },
            'adversarial_analysis': {
                'attack_detected': bool(analysis.get('is_adversarial', False)),
                'defense_applied': bool(analysis.get('is_adversarial', False))
            }
        }
        
        # Clean prediction_result for safety
        prediction_result = clean_numpy_types(prediction_result)
        
        # Save enhanced image as visualization
        if enhanced_img is not None:
            viz_filename = f"enhanced_{filename}"
            viz_filepath = os.path.join(UPLOAD_DIR, viz_filename)
            enhanced_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))
            enhanced_pil.save(viz_filepath)
        else:
            viz_filename = filename
        
        # Save to database
        record = {
            'device_id': str(device),
            'timestamp': int(timestamp),
            'original_image': str(filename),
            'enhanced_image': str(viz_filename),
            'prediction': prediction_result,
            'processing_analysis': analysis
        }
        record = clean_numpy_types(record)
        
        collection.insert_one(record)
        logger.info(f"Disease detected: {predicted_class} ({confidence:.3f})")
        
        return JSONResponse({
            'status': 'success',
            'prediction': prediction_result,
            'original_image': filename,
            'enhanced_image': viz_filename,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)

@app.get('/records')
async def get_detection_records(limit: int = 20):
    """Get recent detection records"""
    try:
        docs = list(collection.find().sort('timestamp', -1).limit(limit))
        for doc in docs:
            doc['_id'] = str(doc['_id'])
        return JSONResponse({'status': 'success', 'records': docs})
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)
@app.delete('/records/{record_id}')
async def delete_record(record_id: str):
    """
    Delete a specific detection record by its MongoDB _id.
    """
    try:
        if not ObjectId.is_valid(record_id):
            return JSONResponse({'status': 'error', 'detail': 'Invalid record ID'}, status_code=400)
        
        # Convert string to ObjectId
        obj_id = ObjectId(record_id)
        
        # Find and delete the record
        result = collection.delete_one({'_id': obj_id})
        
        if result.deleted_count == 1:
            logger.info(f"Record deleted: {record_id}")
            return JSONResponse({'status': 'success', 'detail': 'Record deleted successfully'})
        else:
            return JSONResponse({'status': 'error', 'detail': 'Record not found'}, status_code=404)
            
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)

@app.get('/image/{filename}')
async def serve_image(filename: str):
    """Serve uploaded images"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return JSONResponse({'error': 'Image not found'}, status_code=404)

# Routes for different pages
@app.get('/')
async def home_page():
    """Home page with project overview"""
    return HTMLResponse(content=get_home_page())

@app.get('/predict')
async def prediction_page():
    """Disease prediction page"""
    return HTMLResponse(content=get_prediction_page())

@app.get('/records-page')
async def records_page():
    """Records viewing page"""
    return HTMLResponse(content=get_records_page())

@app.get('/webcam')
async def webcam_page():
    """Real-time webcam detection page"""
    return HTMLResponse(content=get_webcam_page())

def get_home_page() -> str:
    """Generate home page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detection System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0f9b0f 0%, #004d00 100%);  /* dark green shades */
    min-height: 100vh;
    color: #333;
}


        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .nav-links a:hover {
            background: #667eea;
            color: white;
        }
        
        .hero {
            text-align: center;
            padding: 4rem 2rem;
            color: white;
        }
        
        .hero h1 {
            font-size: 2rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            padding: 4rem 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .cta-section {
            text-align: center;
            padding: 4rem 2rem;
            color: white;
        }
        
        .cta-button {
            display: inline-block;
            background: #fff;
            color: #667eea;
            padding: 1rem 2rem;
            text-decoration: none;
            border-radius: 50px;
            font-weight: bold;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .cta-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">🌿 PlantDoc AI</div>
            <ul class="nav-links">
                <li><a href="/">Home</a></li>
                <li><a href="/predict">Predict</a></li>
                <li><a href="/records-page">Records</a></li>
                <li><a href="/webcam">Live Cam</a></li>
            </ul>
        </div>
    </nav>
    
    <section class="hero">
        <h1>AGRICULTURE PLANT DISEASE DETECTION AND TREATMENT PREDCITIONS</h1>
        <p>AI-powered robust defense system disease identification with weather-resistant analysis and adversarial attacks</p>
    </section>
    
    <section class="features">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3>AI Disease Detection</h3>
            <p>Advanced CNN model trained on multiple plant diseases including anthracnose, powdery mildew, yellow mosaic, and leaf crinkle patterns.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🌦️</div>
            <h3>Weather Resistant</h3>
            <p>Automatically adjusts for foggy conditions, low light, and poor weather. Enhanced image processing ensures accurate detection regardless of environmental conditions.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <h3>Adversarial Defense</h3>
            <p>Built-in protection against adversarial attacks and noisy data. Applies defense mechanisms including Gaussian blur and median filtering for reliable results.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">💊</div>
            <h3>Treatment Solutions</h3>
            <p>Comprehensive treatment recommendations including chemical solutions, organic alternatives, and prevention strategies for each detected disease.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📹</div>
            <h3>Real-time Detection</h3>
            <p>Live webcam integration for continuous monitoring. Real-time processing of plant images with instant disease identification and alerts.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Historical Tracking</h3>
            <p>Complete detection history with timestamps, confidence scores, and treatment effectiveness tracking for better agricultural management.</p>
        </div>
    </section>
    
    <section class="cta-section">
        <h2>Start Detecting Plant Diseases Now</h2>
        <p>Upload an image or use live camera feed for instant analysis</p>
        <a href="/predict" class="cta-button">Begin Detection</a>
    </section>
</body>
</html>
    """

def get_prediction_page() -> str:
    """Generate prediction page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disease Prediction - PlantDoc AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .nav-links a:hover, .nav-links a.active {
            background: #667eea;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        @media (max-width: 768px) {
            .main-content { grid-template-columns: 1fr; }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .upload-section h2 {
            color: #4a5568;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .file-input {
            width: 100%;
            padding: 1rem;
            border: 2px dashed #cbd5e0;
            border-radius: 10px;
            background: #f7fafc;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }
        
        .file-input:hover {
            border-color: #667eea;
            background: #e6fffa;
        }
        
        .device-input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .predict-btn {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section h2 {
            color: #4a5568;
            margin-bottom: 1.5rem;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .disease-info {
            background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .disease-name {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 0.5rem;
        }
        
        .confidence-bar {
            background: #e2e8f0;
            border-radius: 10px;
            height: 8px;
            margin: 0.5rem 0;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #48bb78, #38a169);
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .severity {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: bold;
            display: inline-block;
            margin: 0.5rem 0;
        }
        
        .severity.high { background: #fed7d7; color: #c53030; }
        .severity.medium { background: #feebc8; color: #c05621; }
        .severity.none { background: #c6f6d5; color: #276749; }
        
        .treatment-section {
            background: #ebf8ff;
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
        
        .symptoms-list, .solutions-list {
            list-style: none;
            padding-left: 0;
        }
        
        .symptoms-list li, .solutions-list li {
            padding: 0.25rem 0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .symptoms-list li:before {
            content: "🔸 ";
            color: #667eea;
        }
        
        .solutions-list li:before {
            content: "🌿 ";
            color: #48bb78;
        }
        
        .weather-alert {
            background: #fef5e7;
            border: 2px solid #f6ad55;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            color: #c05621;
        }
        
        .loading {
            text-align: center;
            color: #667eea;
            font-style: italic;
            padding: 2rem;
        }
        
        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 1rem;
            border-radius: 5px;
            border: 2px solid #fc8181;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">🌿 PlantDoc AI</div>
            <ul class="nav-links">
                <li><a href="/">Home</a></li>
                <li><a href="/predict" class="active">Predict</a></li>
                <li><a href="/records-page">Records</a></li>
                <li><a href="/webcam">Live Cam</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <div class="main-content">
            <div class="card upload-section">
                <h2>📸 Upload Plant Image</h2>
                <form id="predictForm">
                    <input type="file" id="imageFile" accept="image/*" class="file-input" required>
                    <input type="text" id="deviceId" placeholder="Device/Location ID (optional)" class="device-input">
                    <button type="submit" class="predict-btn" id="predictBtn">
                        🔍 Analyze Plant Health
                    </button>
                </form>
            </div>
            
            <div class="card results-section">
                <h2>🔬 Analysis Results</h2>
                <div id="results">
                    <p style="text-align: center; color: #666; padding: 2rem;">
                        Upload a plant image to see detailed disease analysis and treatment recommendations.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const deviceId = document.getElementById('deviceId').value;
            const resultsDiv = document.getElementById('results');
            const predictBtn = document.getElementById('predictBtn');
            
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }
            
            // Show loading state
            resultsDiv.innerHTML = '<div class="loading">🔄 Analyzing plant image...<br>Processing weather conditions and checking for diseases...</div>';
            predictBtn.disabled = true;
            predictBtn.textContent = '🔄 Processing...';
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            if (deviceId) formData.append('device_id', deviceId);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    // Show user-friendly error for non-plant images
                    if (data.plant_detected === false) {
                        resultsDiv.innerHTML = `
                            <div class="error">
                                <strong>❌ No Plant Detected</strong><br>
                                ${data.detail}<br><br>
                                <small>${data.message}</small>
                            </div>
                        `;
                    } else {
                        resultsDiv.innerHTML = `<div class="error">❌ ${data.detail}</div>`;
                    }
                }
                
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = '🔍 Analyze Plant Health';
            }
        });
        
        function displayResults(data) {
            const pred = data.prediction;
            const analysis = data.analysis;
            
            let html = '';
            
            // Show enhanced image
            if (data.enhanced_image) {
                html += `<img src="/image/${data.enhanced_image}" class="result-image" alt="Enhanced Plant Analysis">`;
            }
            
            // Disease information
            html += `
                <div class="disease-info">
                    <div class="disease-name">${pred.disease_name}</div>
                    <div style="color: #666; margin-bottom: 1rem;">${pred.description}</div>
                    
                    <div style="margin-bottom: 0.5rem;">Confidence: ${(pred.confidence * 100).toFixed(1)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                    </div>
                    
                    <span class="severity ${pred.severity.toLowerCase()}">${pred.severity} Severity</span>
                </div>
            `;
            
            // Weather conditions alert
            if (pred.weather_conditions && (pred.weather_conditions.foggy_detected || pred.weather_conditions.dark_conditions)) {
                html += `
                    <div class="weather-alert">
                        ⚠️ Weather conditions detected: 
                        ${pred.weather_conditions.foggy_detected ? 'Foggy conditions ' : ''}
                        ${pred.weather_conditions.dark_conditions ? 'Low light ' : ''}
                        ${pred.weather_conditions.low_contrast ? 'Low contrast ' : ''}
                        - Image automatically enhanced for better analysis.
                    </div>
                `;
            }
            
            // Adversarial defense notification
            if (pred.adversarial_analysis && pred.adversarial_analysis.attack_detected) {
                html += `
                    <div class="weather-alert" style="background: #fed7d7; border-color: #fc8181; color: #c53030;">
                        🛡️ Potential adversarial attack detected and defended against for accurate results.
                    </div>
                `;
            }
            
            // Symptoms
            if (pred.symptoms && pred.symptoms.length > 0) {
                html += `
                    <div class="treatment-section">
                        <h4>🔍 Symptoms to Look For:</h4>
                        <ul class="symptoms-list">
                            ${pred.symptoms.map(symptom => `<li>${symptom}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            // Treatment
            html += `
                <div class="treatment-section">
                    <h4>💊 Recommended Treatment:</h4>
                    <p>${pred.treatment}</p>
                </div>
            `;
            
            // Prevention
            html += `
                <div class="treatment-section">
                    <h4>🛡️ Prevention Methods:</h4>
                    <p>${pred.prevention}</p>
                </div>
            `;
            
            // Organic solutions
            if (pred.organic_solutions && pred.organic_solutions.length > 0) {
                html += `
                    <div class="treatment-section">
                        <h4>🌿 Organic Solutions:</h4>
                        <ul class="solutions-list">
                            ${pred.organic_solutions.map(solution => `<li>${solution}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
    """

def get_records_page() -> str:
    """Generate records page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Records - PlantDoc AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .nav-links a:hover, .nav-links a.active {
            background: #667eea;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .page-header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        
        .page-header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .records-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
        }
        
        .record-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .record-card:hover {
            transform: translateY(-5px);
        }
        
        .record-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .disease-name {
            font-size: 1.25rem;
            font-weight: bold;
            color: #2d3748;
        }
        
        .confidence-badge {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.875rem;
        }
        
        .record-details {
            color: #666;
            font-size: 0.875rem;
            line-height: 1.5;
        }
        
        .record-meta {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #888;
            font-size: 0.75rem;
        }
        .delete-btn {
    background: #fc8181;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background 0.3s ease;
    margin-top: 0.5rem;
}

.delete-btn:hover {
    background: #ef4444;
}

.record-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    position: relative;  /* Add this for button positioning */
}

.record-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.delete-btn-container {
    display: flex;
    justify-content: flex-end;
    margin-top: auto;  /* pushes it to the bottom */
}

        .no-records {
            text-align: center;
            color: white;
            padding: 4rem 2rem;
        }
        
        .loading {
            text-align: center;
            color: white;
            padding: 4rem 2rem;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">🌿 PlantDoc AI</div>
            <ul class="nav-links">
                <li><a href="/">Home</a></li>
                <li><a href="/predict">Predict</a></li>
                <li><a href="/records-page" class="active">Records</a></li>
                <li><a href="/webcam">Live Cam</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <div class="page-header">
            <h1>📊 Detection Records</h1>
            <p>Complete history of plant disease detections and analysis</p>
        </div>
        
        <div id="recordsContainer">
            <div class="loading">Loading detection records...</div>
        </div>
    </div>

    <script>
        async function loadRecords() {
            try {
                const response = await fetch('/records');
                const data = await response.json();
                
                if (data.status === 'success' && data.records.length > 0) {
                    displayRecords(data.records);
                } else {
                    document.getElementById('recordsContainer').innerHTML = `
                        <div class="no-records">
                            <h2>No Records Yet</h2>
                            <p>Start by analyzing some plant images to build your detection history.</p>
                            <a href="/predict" style="color: white; text-decoration: underline;">Begin Analysis</a>
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('recordsContainer').innerHTML = `
                    <div class="no-records">
                        <h2>Failed to Load Records</h2>
                        <p>Please check your connection and try again.</p>
                    </div>
                `;
            }
        }
        
       function displayRecords(records) {
    const container = document.getElementById('recordsContainer');
    
    const recordsHTML = records.map(record => {
        const pred = record.prediction;
        const timestamp = new Date(record.timestamp * 1000).toLocaleString();
        
        return `
            <div class="record-card" data-record-id="${record._id}">
                <img src="/image/${record.enhanced_image || record.original_image}" 
                     class="record-image" alt="Plant Analysis">
                
                <div class="record-header">
                    <div class="disease-name">${pred.disease_name}</div>
                    <div class="confidence-badge">${(pred.confidence * 100).toFixed(1)}%</div>
                </div>
                
                <div class="record-details">
                    <strong>Severity:</strong> ${pred.severity}<br>
                    <strong>Treatment:</strong> ${pred.treatment.substring(0, 100)}...
                    ${record.processing_analysis && record.processing_analysis.is_adversarial ? 
                        '<br><span style="color: #c53030;">🛡️ Adversarial defense applied</span>' : ''}
                    ${record.processing_analysis && record.processing_analysis.is_foggy ? 
                        '<br><span style="color: #c05621;">🌫️ Weather enhancement applied</span>' : ''}
                </div>
                
                <div class="record-meta">
                    Device: ${record.device_id} | Detected: ${timestamp}
                </div>

                <div class="delete-btn-container">
                    <button class="delete-btn" onclick="deleteRecord('${record._id}')">🗑️ Delete</button>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = `<div class="records-grid">${recordsHTML}</div>`;
}


// New delete function (add this after displayRecords)
async function deleteRecord(recordId) {
    if (!confirm('Are you sure you want to delete this record? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/records/${recordId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert('Record deleted successfully!');
            loadRecords();  // Reload the records list
        } else {
            alert(`Error: ${data.detail}`);
        }
    } catch (error) {
        alert(`Network error: ${error.message}`);
    }
}

        // Load records on page load
        loadRecords();
        
        // Auto-refresh every 30 seconds
        setInterval(loadRecords, 30000);
    </script>
</body>
</html>
    """

def get_webcam_page() -> str:
    """Generate webcam page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Webcam Detection - PlantDoc AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        
        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .nav-links {
            display: flex;
            list-style: none;
            gap: 2rem;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .nav-links a:hover, .nav-links a.active {
            background: #667eea;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .page-header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        
        .webcam-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        @media (max-width: 768px) {
            .webcam-section { grid-template-columns: 1fr; }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .webcam-container {
            text-align: center;
        }
        
        #video, #canvas {
            width: 100%;
            max-width: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .controls {
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .live-results {
            min-height: 300px;
        }
        
        .auto-detect-info {
            background: #e6fffa;
            border: 2px solid #4fd1c7;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            color: #234e52;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">🌿 PlantDoc AI</div>
            <ul class="nav-links">
                <li><a href="/">Home</a></li>
                <li><a href="/predict">Predict</a></li>
                <li><a href="/records-page">Records</a></li>
                <li><a href="/webcam" class="active">Live Cam</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <div class="page-header">
            <h1>📹 Live Webcam Detection</h1>
            <p>Real-time plant disease detection using your camera</p>
        </div>
        
        <div class="webcam-section">
            <div class="card">
                <h2>📷 Live Camera Feed</h2>
                <div class="webcam-container">
                    <video id="video" autoplay muted style="display: none;"></video>
                    <canvas id="canvas" style="display: none;"></canvas>
                    <div id="camera-placeholder" style="padding: 2rem; background: #f7fafc; border: 2px dashed #cbd5e0; border-radius: 10px;">
                        <p>Click "Start Camera" to begin live detection</p>
                    </div>
                </div>
                
                <div class="controls">
                    <button id="startCamera" class="btn btn-primary">📷 Start Camera</button>
                    <button id="stopCamera" class="btn btn-warning" style="display: none;">⏹️ Stop Camera</button>
                    <button id="captureBtn" class="btn btn-success" style="display: none;">📸 Analyze Current Frame</button>
                </div>
                
                <div class="auto-detect-info" id="autoDetectInfo" style="display: none;">
                    <strong>🔄 Auto-Detection Active</strong><br>
                    Camera feed is being analyzed automatically every 5 seconds for plant diseases.
                </div>
            </div>
            
            <div class="card live-results">
                <h2>🔬 Live Analysis Results</h2>
                <div id="liveResults">
                    <p style="text-align: center; color: #666; padding: 2rem;">
                        Start the camera to see real-time plant disease detection results.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let video, canvas, context;
let stream = null;
let isAnalyzing = false;
let autoDetectInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    context = canvas.getContext('2d');
    
    document.getElementById('startCamera').addEventListener('click', startCamera);
    document.getElementById('stopCamera').addEventListener('click', stopCamera);
    document.getElementById('captureBtn').addEventListener('click', captureAndAnalyze);
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'environment'
            } 
        });
        
        video.srcObject = stream;
        video.style.display = 'block';
        document.getElementById('camera-placeholder').style.display = 'none';
        
        // Show controls (FIXED: added .style)
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-block';
        document.getElementById('captureBtn').style.display = 'inline-block';
        document.getElementById('autoDetectInfo').style.display = 'block';
        
        // Set up canvas
        video.addEventListener('loadedmetadata', function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        });
        
        // Start auto-detection every 5 seconds
        autoDetectInterval = setInterval(autoDetectFrame, 5000);
        
    } catch (error) {
        alert('Could not access camera: ' + error.message);
        console.error('Camera error:', error);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (autoDetectInterval) {
        clearInterval(autoDetectInterval);
        autoDetectInterval = null;
    }
    
    video.style.display = 'none';
    canvas.style.display = 'none';
    document.getElementById('camera-placeholder').style.display = 'block';
    document.getElementById('camera-placeholder').innerHTML = '<p>Camera stopped. Click "Start Camera" to begin again.</p>';
    
    // Reset controls
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('stopCamera').style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('autoDetectInfo').style.display = 'none';
}

async function captureAndAnalyze() {
    if (isAnalyzing || !stream) return;
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(async (blob) => {
        await analyzeFrame(blob, false);
    }, 'image/jpeg', 0.8);
}

async function autoDetectFrame() {
    if (isAnalyzing || !stream) return;
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(async (blob) => {
        await analyzeFrame(blob, true);
    }, 'image/jpeg', 0.6);
}

async function analyzeFrame(imageBlob, isAuto) {
    if (isAnalyzing) return;
    
    isAnalyzing = true;
    const resultsDiv = document.getElementById('liveResults');
    
    if (!isAuto) {
        resultsDiv.innerHTML = '<div style="text-align: center; padding: 1rem; color: #667eea;">🔄 Analyzing current frame...</div>';
    }
    
    try {
        const formData = new FormData();
        formData.append('file', imageBlob, 'webcam_frame.jpg');
        formData.append('device_id', 'webcam_' + Date.now());
        
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayLiveResults(data, isAuto);
        } else {
            if (data.plant_detected === false) {
                resultsDiv.innerHTML = `
                    <div style="background: #fef5e7; border: 2px solid #f6ad55; border-radius: 8px; padding: 1rem; color: #c05621;">
                        <strong>🌿 Plant Not Detected</strong><br>
                        ${data.detail}<br><br>
                        <small>${data.message}</small>
                    </div>
                `;
            } else {
                resultsDiv.innerHTML = `<div style="color: #c53030; padding: 1rem;">❌ ${data.detail}</div>`;
            }
        }
        
    } catch (error) {
        if (!isAuto) {
            resultsDiv.innerHTML = `<div style="color: #c53030; padding: 1rem;">❌ Network error: ${error.message}</div>`;
        }
    } finally {
        isAnalyzing = false;
    }
}

// ADD THIS FUNCTION - IT WAS MISSING!
function displayLiveResults(data, isAuto) {
    const pred = data.prediction;
    const timestamp = new Date().toLocaleTimeString();
    
    let html = `
        <div style="text-align: center; margin-bottom: 1rem;">
            <strong style="color: #2d3748; font-size: 1.2rem;">${pred.disease_name}</strong>
            <div style="color: #667eea; margin: 0.5rem 0;">Confidence: ${(pred.confidence * 100).toFixed(1)}%</div>
            <div style="font-size: 0.9rem; color: #666;">Last updated: ${timestamp} ${isAuto ? '(Auto)' : '(Manual)'}</div>
        </div>
        
        <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong>Severity:</strong> ${pred.severity}<br>
            <strong>Description:</strong> ${pred.description}
        </div>
    `;
    
    if (pred.weather_conditions && (pred.weather_conditions.foggy_detected || pred.weather_conditions.dark_conditions)) {
        html += `
            <div style="background: #fef5e7; border: 2px solid #f6ad55; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem; font-size: 0.9rem;">
                🌦️ Weather conditions detected - image enhanced for better analysis
            </div>
        `;
    }
    
    if (pred.adversarial_analysis && pred.adversarial_analysis.attack_detected) {
        html += `
            <div style="background: #fed7d7; border: 2px solid #fc8181; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem; font-size: 0.9rem;">
                🛡️ Noise detected and filtered for accurate results
            </div>
        `;
    }
    
    html += `
        <div style="background: #ebf8ff; border-left: 4px solid #667eea; padding: 1rem; border-radius: 5px;">
            <strong>Quick Treatment:</strong><br>
            ${pred.treatment.substring(0, 120)}${pred.treatment.length > 120 ? '...' : ''}
        </div>
    `;
    
    document.getElementById('liveResults').innerHTML = html;
}
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)