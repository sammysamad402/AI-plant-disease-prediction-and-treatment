from fastapi import FastAPI, File, UploadFile, Form, Request
from dotenv import load_dotenv
load_dotenv()
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
from consultation import consultation_router
from diary import diary_router
from crop_calendar import calendar_router
import requests
import threading
from shared_ui import shared_head, build_nav, TOAST_SCRIPT

import os
import gdown

# Auto-download model if not present
MODEL_PATH = os.environ.get("MODEL_PATH", "BPLD_CNN_model_v3.h5")
if not os.path.exists(MODEL_PATH):
    drive_id = os.environ.get("MODEL_DRIVE_ID", "")
    if drive_id:
        print("📥 Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={drive_id}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_numpy_types(obj):
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
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        return clean_numpy_types(obj.item())
    else:
        return obj


app = FastAPI(title="Advanced Plant Disease Detection System")
app.include_router(consultation_router)
app.include_router(diary_router)
app.include_router(calendar_router)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.get_database("plant_disease_db")
collection = db.get_collection("detections")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['anthracnose', 'healthy', 'leaf_crinkle', 'powdery_mildew', 'yellow_mosaic']

DISEASE_INFO = {
    'anthracnose': {
        'severity': 'High',
        'color': '#E74C3C',
        'description': 'Fungal disease causing dark lesions on leaves and stems',
        'symptoms': ['Dark, sunken spots on leaves', 'Brown or black lesions', 'Leaf yellowing and drop'],
        'treatment': 'Apply copper-based fungicides, remove infected plant parts, ensure good air circulation',
        'prevention': 'Avoid overhead watering, plant resistant varieties, maintain proper spacing',
        'organic_solutions': ['Neem oil spray', 'Baking soda solution', 'Compost tea application']
    },
    'healthy': {
        'severity': 'None',
        'color': '#27AE60',
        'description': 'Plant appears healthy with no visible disease symptoms',
        'symptoms': ['Green, vibrant foliage', 'No discoloration or lesions', 'Normal growth patterns'],
        'treatment': 'No treatment needed — maintain current care practices',
        'prevention': 'Continue regular monitoring, maintain optimal growing conditions',
        'organic_solutions': ['Regular inspection', 'Balanced nutrition', 'Proper watering schedule']
    },
    'leaf_crinkle': {
        'severity': 'Medium',
        'color': '#E67E22',
        'description': 'Viral disease causing leaf distortion and reduced photosynthesis',
        'symptoms': ['Wrinkled or puckered leaves', 'Stunted growth', 'Yellowing between veins'],
        'treatment': 'Remove affected leaves immediately, control insect vectors, quarantine infected plants',
        'prevention': 'Use virus-free planting material, control aphids and whiteflies, maintain plant hygiene',
        'organic_solutions': ['Reflective mulch', 'Beneficial insect habitat', 'Quarantine new plants']
    },
    'powdery_mildew': {
        'severity': 'Medium',
        'color': '#F39C12',
        'description': 'Fungal disease creating white powdery coating on leaf surfaces',
        'symptoms': ['White powdery spots on leaves', 'Leaf curling and distortion', 'Reduced plant vigor'],
        'treatment': 'Apply sulfur-based fungicides, improve air circulation, reduce humidity around plants',
        'prevention': 'Avoid overhead watering, ensure proper plant spacing, choose resistant varieties',
        'organic_solutions': ['Milk spray (1:10 ratio)', 'Baking soda solution', 'Horticultural oils']
    },
    'yellow_mosaic': {
        'severity': 'High',
        'color': '#F1C40F',
        'description': 'Viral disease causing yellow mottling and severe yield reduction',
        'symptoms': ['Yellow patches on leaves', 'Mosaic-like patterns', 'Stunted plant growth'],
        'treatment': 'Remove infected plants immediately, control whitefly vectors, use resistant varieties',
        'prevention': 'Monitor for whiteflies, use yellow sticky traps, plant virus-resistant cultivars',
        'organic_solutions': ['Vector control with beneficial insects', 'Reflective mulches', 'Crop rotation']
    }
}

MODEL_PATH = os.environ.get("MODEL_PATH", "BPLD_CNN_model.h5")
model = None
try:
    if os.path.exists(MODEL_PATH):
        import h5py
        with h5py.File(MODEL_PATH, 'r+') as f:
            model_config = f.attrs.get('model_config')
            if model_config:
                import json
                config = json.loads(model_config)
                config_str = json.dumps(config).replace(
                    '"batch_shape"', '"batch_input_shape"'
                )
                f.attrs['model_config'] = config_str.encode()
        model = keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    logger.error(f'Model load error: {e}')

def fgsm_attack(model, image, epsilon=0.03):
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.reduce_max(prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    perturbed = image + epsilon * signed_grad
    return tf.clip_by_value(perturbed, 0, 1)


def pgd_attack(model, image, epsilon=0.03, alpha=0.01, iters=5):
    original = tf.cast(tf.identity(image), tf.float32)
    perturbed = tf.cast(tf.identity(image), tf.float32)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(perturbed)
            prediction = model(perturbed)
            loss = tf.reduce_max(prediction)
        gradient = tape.gradient(loss, perturbed)
        perturbed = perturbed + alpha * tf.sign(gradient)
        eta = tf.clip_by_value(perturbed - original, -epsilon, epsilon)
        perturbed = tf.clip_by_value(original + eta, 0, 1)
    return perturbed


def cw_attack(model, image, c=1e-4, iters=20, lr=0.01):
    perturbed = tf.Variable(tf.cast(image, tf.float32))
    optimizer = tf.keras.optimizers.Adam(lr)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            prediction = model(perturbed)
            target = tf.argmax(prediction[0])
            loss = -prediction[0][target]
            l2 = tf.norm(perturbed - image)
            total_loss = l2 + c * loss
        grads = tape.gradient(total_loss, perturbed)
        optimizer.apply_gradients([(grads, perturbed)])
        perturbed.assign(tf.clip_by_value(perturbed, 0, 1))
    return perturbed


class WeatherEnhancer:
    @staticmethod
    def detect_weather_conditions(image_array):
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image_array * 255).astype(np.uint8)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        return {
            'is_foggy': bool(mean_intensity > 180 and contrast < 0.3),
            'is_dark': bool(mean_intensity < 80),
            'is_low_contrast': bool(contrast < 0.2),
            'mean_intensity': float(mean_intensity),
            'contrast': float(contrast)
        }

    @staticmethod
    def enhance_for_weather(image_array, weather_conditions):
        pil_image = Image.fromarray((image_array.copy() * 255).astype(np.uint8))
        if weather_conditions['is_foggy']:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(1.5)
            pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.2)
        if weather_conditions['is_dark']:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(1.3)
        if weather_conditions['is_low_contrast']:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(1.4)
        return np.array(pil_image).astype('float32') / 255.0


class AdversarialDefense:
    @staticmethod
    def detect_adversarial_patterns(image_array):
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image_array * 255).astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        high_freq_energy = np.mean(gradient_magnitude)
        gradient_variance = np.var(gradient_magnitude)
        noise_threshold = 40
        is_adversarial = high_freq_energy > noise_threshold and gradient_variance > 5000
        return {
            'is_adversarial': bool(is_adversarial),
            'confidence': float(min(high_freq_energy / noise_threshold, 1.0)),
            'high_freq_energy': float(high_freq_energy),
            'gradient_variance': float(gradient_variance)
        }

    @staticmethod
    def apply_defense_mechanisms(image_array):
        defended = image_array.copy()
        defended = cv2.GaussianBlur(defended, (3, 3), 0.8)
        defended_uint8 = (defended * 255).astype(np.uint8)
        defended_uint8 = cv2.medianBlur(defended_uint8, 3)
        return defended_uint8.astype('float32') / 255.0


class PlantDetector:
    @staticmethod
    def is_plant_image(image_array):
        image_uint8 = (image_array * 255).astype(np.uint8)
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
        green_mask = cv2.bitwise_or(mask1, mask2)
        green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        color_variation = (np.std(image_array[:,:,0]) + np.std(image_array[:,:,1]) + np.std(image_array[:,:,2])) / 3
        is_plant = False
        confidence = 0.0
        reason = "Not a plant"
        if green_percentage > 15 and 5 < edge_density < 40 and color_variation > 0.05:
            is_plant = True
            confidence = min((green_percentage / 50) * 100, 100)
            reason = "Plant detected based on green content and texture"
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        skin_percentage = (np.sum(skin_mask > 0) / skin_mask.size) * 100
        if skin_percentage > 30:
            is_plant = False
            confidence = 0.0
            reason = "Human skin detected — not a plant"
        return {
            'is_plant': bool(is_plant),
            'confidence': float(confidence),
            'green_percentage': float(green_percentage),
            'edge_density': float(edge_density),
            'skin_detected': bool(skin_percentage > 30),
            'reason': reason
        }


def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img.resize(target_size)).astype('float32') / 255.0
        plant_check = PlantDetector.is_plant_image(img_array)
        if not plant_check['is_plant']:
            return None, None, {'plant_detected': False, 'reason': plant_check['reason']}
        weather_enhancer = WeatherEnhancer()
        weather_conditions = weather_enhancer.detect_weather_conditions(img_array)
        if any(weather_conditions[k] for k in ['is_foggy', 'is_dark', 'is_low_contrast']):
            img_array = weather_enhancer.enhance_for_weather(img_array, weather_conditions)
        defense_system = AdversarialDefense()
        adv_analysis = defense_system.detect_adversarial_patterns(img_array)
        if adv_analysis['is_adversarial']:
            img_array = defense_system.apply_defense_mechanisms(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        full_analysis = {**plant_check, **adv_analysis, **weather_conditions, 'plant_detected': True}
        return img_batch, img_array, full_analysis
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        return None, None, None


@app.post('/detect')
async def detect_disease(file: UploadFile = File(...), device_id: str = Form(None)):
    try:
        if model is None:
            return JSONResponse({'status': 'error', 'detail': 'Model not loaded'}, status_code=500)
        contents = await file.read()
        timestamp = int(time.time())
        device = device_id or "device"
        filename = f"{device}_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(contents)
        processed_img, enhanced_img, analysis = preprocess_image(filepath)
        if analysis and not analysis.get('plant_detected', False):
            return JSONResponse({
                'status': 'error',
                'detail': analysis.get('reason', 'No plant detected in image'),
                'message': 'Please upload an image containing plant leaves for disease detection',
                'plant_detected': False
            }, status_code=400)
        if processed_img is None:
            return JSONResponse({'status': 'error', 'detail': 'Failed to process image'}, status_code=400)
        analysis = clean_numpy_types(analysis or {})
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        processed_img = (processed_img - imagenet_mean) / imagenet_std
        predictions = model.predict(processed_img, verbose=0)
        try:
            fgsm_img = fgsm_attack(model, processed_img)
            pgd_img = pgd_attack(model, processed_img)
            cw_img = cw_attack(model, processed_img)
            fgsm_class = CLASS_NAMES[np.argmax(model.predict(fgsm_img, verbose=0)[0])]
            pgd_class = CLASS_NAMES[np.argmax(model.predict(pgd_img, verbose=0)[0])]
            cw_class = CLASS_NAMES[np.argmax(model.predict(cw_img, verbose=0)[0])]
        except Exception as e:
            logger.error(f"Attack simulation failed: {e}")
            fgsm_class = pgd_class = cw_class = None
        probs = predictions[0]
        sorted_idx = np.argsort(probs)[::-1]
        top1 = CLASS_NAMES[sorted_idx[0]]
        top2 = CLASS_NAMES[sorted_idx[1]] if len(sorted_idx) > 1 else None
        top1_conf = float(probs[sorted_idx[0]])
        top2_conf = float(probs[sorted_idx[1]]) if top2 else 0.0
        if top1 == "healthy" and top2_conf > 0.10:
            top1, top1_conf = top2, top2_conf
        if top2 == top1 or top2_conf < 0.01:
            top2 = None
        if top1_conf >= 0.98:
            interpretation = "Very high confidence prediction"
        elif top1_conf >= 0.80:
            interpretation = "High confidence prediction"
        elif top1_conf >= 0.60:
            interpretation = "Moderate confidence prediction"
        else:
            interpretation = "Low confidence — result may be unreliable"
        predicted_class = top1
        confidence = top1_conf
        # ── Confidence Warning ──
        warning = None
        if confidence < 0.6:
            warning = "⚠ Low confidence — result may be inaccurate"

# ── Explanation (NEW FEATURE ONLY) ──
        explanations = {
        "powdery_mildew": "Detected white powdery fungal texture on leaves.",
        "anthracnose": "Detected dark lesions and infection spots.",
        "leaf_crinkle": "Detected distorted and wrinkled leaf structure.",
         "yellow_mosaic": "Detected yellow mosaic pattern on leaves.",
        "healthy": "Leaf structure appears normal with no disease patterns."
        }

        explanation = explanations.get(predicted_class, "Pattern detected by CNN model")
        disease_data = DISEASE_INFO.get(predicted_class, {})
        prediction_result = {
            'warning': warning,
            'explanation': explanation,
            'disease_name': predicted_class.replace('_', ' ').title(),
            'confidence': confidence,
            'severity': disease_data.get('severity', 'Unknown'),
            'description': disease_data.get('description', 'No description available'),
            'symptoms': disease_data.get('symptoms', []),
            'treatment': disease_data.get('treatment', 'Consult agricultural expert'),
            'prevention': disease_data.get('prevention', 'Follow general plant care practices'),
            'organic_solutions': disease_data.get('organic_solutions', []),
            'smart_analysis': {
                'primary': top1, 'primary_confidence': top1_conf,
                'secondary': top2, 'secondary_confidence': top2_conf,
                'interpretation': interpretation
            },
            'adversarial_tests': {
                'fgsm_prediction': fgsm_class,
                'pgd_prediction': pgd_class,
                'cw_prediction': cw_class
            },
            'weather_conditions': {
                'foggy_detected': bool(analysis.get('is_foggy', False)),
                'dark_conditions': bool(analysis.get('is_dark', False)),
                'low_contrast': bool(analysis.get('is_low_contrast', False))
            },
            'adversarial_analysis': {
                'attack_detected': bool(analysis.get('is_adversarial', False)),
                'defense_applied': bool(analysis.get('is_adversarial', False))
            },
            'overconfidence_warning': "Model is highly confident. Similar diseases may still be misclassified." if top1_conf > 0.98 else None
        }
        prediction_result = clean_numpy_types(prediction_result)
        if enhanced_img is not None:
            viz_filename = f"enhanced_{filename}"
            Image.fromarray((enhanced_img * 255).astype(np.uint8)).save(os.path.join(UPLOAD_DIR, viz_filename))
        else:
            viz_filename = filename
        record = clean_numpy_types({
            'device_id': str(device),
            'timestamp': int(timestamp),
            'original_image': str(filename),
            'enhanced_image': str(viz_filename),
            'prediction': prediction_result,
            'processing_analysis': analysis
        })
        collection.insert_one(record)
        logger.info(f"Detected: {predicted_class} ({confidence:.3f})")
        return JSONResponse({
            'status': 'success',
            'prediction': prediction_result,
            'original_image': filename,
            'enhanced_image': viz_filename,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)


@app.get('/records')
async def get_detection_records(limit: int = 1000):
    try:
        docs = list(collection.find().sort('timestamp', -1))
        for doc in docs:
            doc['_id'] = str(doc['_id'])
        return JSONResponse({'status': 'success', 'records': docs})
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)


@app.delete('/records/{record_id}')
async def delete_record(record_id: str):
    try:
        if not ObjectId.is_valid(record_id):
            return JSONResponse({'status': 'error', 'detail': 'Invalid record ID'}, status_code=400)
        result = collection.delete_one({'_id': ObjectId(record_id)})
        if result.deleted_count == 1:
            return JSONResponse({'status': 'success', 'detail': 'Record deleted successfully'})
        return JSONResponse({'status': 'error', 'detail': 'Record not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)


@app.get('/image/{filename}')
async def serve_image(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return JSONResponse({'error': 'Image not found'}, status_code=404)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get('/')
async def home_page():
    return HTMLResponse(content=get_home_page())

@app.get('/predict')
async def prediction_page():
    return HTMLResponse(content=get_prediction_page())

@app.get('/records-page')
async def records_page():
    return HTMLResponse(content=get_records_page())

@app.get('/webcam')
async def webcam_page():
    return HTMLResponse(content=get_webcam_page())


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_home_page() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Home")}<style>
/* home-only styles */
.home-hero{{padding:4rem 1.5rem 3.5rem;text-align:center;
  background:linear-gradient(135deg,var(--p0) 0%,var(--p1) 50%,var(--p2) 100%);
  color:#fff;position:relative;overflow:hidden;}}
.home-hero::before{{content:'';position:absolute;inset:0;
  background:url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M40 0C17.9 0 0 17.9 0 40s17.9 40 40 40 40-17.9 40-40S62.1 0 40 0zm0 70C23.4 70 10 56.6 10 40S23.4 10 40 10s30 13.4 30 30-13.4 30-30 30z'/%3E%3C/g%3E%3C/svg%3E");}}
.home-hero-inner{{max-width:760px;margin:0 auto;position:relative;}}
.hero-eyebrow{{font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  opacity:.8;margin-bottom:.75rem;}}
.home-hero h1{{font-size:2.75rem;font-weight:800;line-height:1.15;margin-bottom:1rem;letter-spacing:-.5px;}}
.home-hero h1 em{{font-style:normal;color:var(--ac);}}
.home-hero .lead{{font-size:1.05rem;opacity:.9;max-width:580px;margin:0 auto 2rem;}}
.hero-cta{{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;}}
.hero-cta .btn{{padding:.85rem 2rem;font-size:.95rem;}}
.btn-white{{background:#fff;color:var(--p1);}}
.btn-white:hover{{background:var(--bg2);transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.15);}}
.btn-outline-white{{background:transparent;color:#fff;border:2px solid rgba(255,255,255,.6);}}
.btn-outline-white:hover{{background:rgba(255,255,255,.12);border-color:#fff;}}

.stats-strip{{background:var(--p0);padding:1.1rem 1.5rem;}}
.stats-strip-inner{{max-width:1320px;margin:0 auto;display:grid;
  grid-template-columns:repeat(5,1fr);gap:1rem;text-align:center;}}
.strip-num{{font-size:1.6rem;font-weight:800;color:#fff;line-height:1;}}
.strip-lbl{{font-size:.72rem;color:rgba(255,255,255,.6);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;margin-top:.2rem;}}
@media(max-width:700px){{.stats-strip-inner{{grid-template-columns:repeat(3,1fr);}}
  .home-hero h1{{font-size:2rem;}}}}

.features-section{{padding:3.5rem 1.5rem;}}
.feat-grid{{max-width:1320px;margin:0 auto;display:grid;
  grid-template-columns:repeat(3,1fr);gap:1.5rem;}}
@media(max-width:900px){{.feat-grid{{grid-template-columns:1fr 1fr;}}}}
@media(max-width:600px){{.feat-grid{{grid-template-columns:1fr;}}}}
.feat-card{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  padding:1.75rem;box-shadow:var(--sh);transition:all .2s;}}
.feat-card:hover{{transform:translateY(-4px);box-shadow:var(--shm);border-color:var(--p3);}}
.feat-icon{{width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,var(--p2),var(--p1));
  display:flex;align-items:center;justify-content:center;font-size:1.5rem;margin-bottom:1rem;}}
.feat-title{{font-size:1rem;font-weight:700;color:var(--tx);margin-bottom:.4rem;}}
.feat-desc{{font-size:.85rem;color:var(--tx2);line-height:1.6;}}

.how-section{{background:var(--p0);padding:3.5rem 1.5rem;color:#fff;}}
.how-inner{{max-width:1100px;margin:0 auto;}}
.how-title{{text-align:center;font-size:1.75rem;font-weight:800;margin-bottom:.5rem;}}
.how-sub{{text-align:center;opacity:.8;margin-bottom:2.5rem;font-size:.95rem;}}
.how-steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;}}
@media(max-width:800px){{.how-steps{{grid-template-columns:1fr 1fr;}}}}
@media(max-width:500px){{.how-steps{{grid-template-columns:1fr;}}}}
.how-step{{text-align:center;padding:1.5rem 1rem;background:rgba(255,255,255,.08);
  border-radius:var(--r);border:1px solid rgba(255,255,255,.12);}}
.step-num{{width:40px;height:40px;border-radius:50%;background:var(--ac);color:#fff;
  font-weight:800;font-size:1rem;display:flex;align-items:center;justify-content:center;
  margin:0 auto 1rem;}}
.step-title{{font-size:.9rem;font-weight:700;margin-bottom:.4rem;}}
.step-desc{{font-size:.8rem;opacity:.8;line-height:1.5;}}

.disease-section{{padding:3.5rem 1.5rem;}}
.disease-inner{{max-width:1100px;margin:0 auto;}}
.diseases-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-top:2rem;}}
@media(max-width:900px){{.diseases-grid{{grid-template-columns:1fr 1fr;}}}}
.disease-card{{border-radius:var(--r);border:2px solid;padding:1.25rem;text-align:center;transition:all .2s;}}
.disease-card:hover{{transform:translateY(-3px);box-shadow:var(--shm);}}
.dis-icon{{font-size:2rem;margin-bottom:.6rem;}}
.dis-name{{font-size:.85rem;font-weight:700;margin-bottom:.3rem;}}
.dis-sev{{font-size:.72rem;font-weight:700;padding:.15rem .5rem;border-radius:20px;}}

.cta-section{{background:linear-gradient(135deg,var(--ac2) 0%,var(--ac) 100%);
  padding:4rem 1.5rem;text-align:center;color:#fff;}}
.cta-section h2{{font-size:2rem;font-weight:800;margin-bottom:.75rem;}}
.cta-section p{{font-size:1rem;opacity:.9;margin-bottom:2rem;}}
</style>
</head>
<body>
{build_nav("/")}

<!-- HERO -->
<section class="home-hero">
  <div class="home-hero-inner">
    <div class="hero-eyebrow">🌱 AI-Powered Agriculture Platform</div>
    <h1>Protect Your Crops with<br><em>Smart Plant Disease Detection</em></h1>
    <p class="lead">Upload a leaf photo or use live camera — get instant diagnosis, treatment plans, and expert advice in multiple Indian languages.</p>
    <div class="hero-cta">
      <a href="/predict" class="btn btn-white btn-lg">🔬 Detect Disease Now</a>
      <a href="/webcam" class="btn btn-outline-white btn-lg">📹 Live Camera</a>
    </div>
  </div>
</section>

<!-- STATS STRIP -->
<div class="stats-strip">
  <div class="stats-strip-inner">
    <div><div class="strip-num">5</div><div class="strip-lbl">Diseases Detected</div></div>
    <div><div class="strip-num">CNN</div><div class="strip-lbl">AI Model</div></div>
    <div><div class="strip-num">3</div><div class="strip-lbl">Attack Defenses</div></div>
    <div><div class="strip-num">9</div><div class="strip-lbl">Languages</div></div>
    <div><div class="strip-num">24/7</div><div class="strip-lbl">Monitoring</div></div>
  </div>
</div>

<!-- FEATURES -->
<section class="features-section">
  <div class="feat-grid">
    <div class="feat-card">
      <div class="feat-icon">🔬</div>
      <div class="feat-title">AI Disease Detection</div>
      <p class="feat-desc">CNN model trained on 5 major plant diseases: anthracnose, powdery mildew, yellow mosaic, leaf crinkle, and healthy leaves.</p>
    </div>
    <div class="feat-card">
      <div class="feat-icon">🌦️</div>
      <div class="feat-title">Weather-Resistant Analysis</div>
      <p class="feat-desc">Auto-detects foggy, dark, and low-contrast conditions and enhances images for accurate diagnosis in any environment.</p>
    </div>
    <div class="feat-card">
      <div class="feat-icon">🛡️</div>
      <div class="feat-title">Adversarial Defense</div>
      <p class="feat-desc">Built-in FGSM, PGD, and C&W attack simulation with Gaussian blur and median filtering to protect results from manipulation.</p>
    </div>
    <div class="feat-card">
      <div class="feat-icon">💬</div>
      <div class="feat-title">Expert Consultation</div>
      <p class="feat-desc">Chat with AgriDoc AI in 9 Indian languages. Upload photos, get voice replies, and save consultations to your Farm Diary.</p>
    </div>
    <div class="feat-card">
      <div class="feat-icon">📔</div>
      <div class="feat-title">Farm Diary</div>
      <p class="feat-desc">Keep a complete record of all detections and expert consultations. Tag, filter, and track your farm's health history over time.</p>
    </div>
    <div class="feat-card">
      <div class="feat-icon">🗓️</div>
      <div class="feat-title">Crop Calendar</div>
      <p class="feat-desc">Month-by-month farming guide for major Indian crops — sowing, irrigation, fertilization, and harvest scheduling made simple.</p>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section class="how-section">
  <div class="how-inner">
    <h2 class="how-title">How It Works</h2>
    <p class="how-sub">Get a complete plant health report in under 10 seconds</p>
    <div class="how-steps">
      <div class="how-step">
        <div class="step-num">1</div>
        <div class="step-title">Upload Photo</div>
        <p class="step-desc">Snap a photo of the affected leaf or use your webcam for live analysis</p>
      </div>
      <div class="how-step">
        <div class="step-num">2</div>
        <div class="step-title">AI Processing</div>
        <p class="step-desc">CNN model analyzes with weather correction and adversarial defense applied</p>
      </div>
      <div class="how-step">
        <div class="step-num">3</div>
        <div class="step-title">Get Diagnosis</div>
        <p class="step-desc">See disease name, confidence score, severity, and smart analysis interpretation</p>
      </div>
      <div class="how-step">
        <div class="step-num">4</div>
        <div class="step-title">Take Action</div>
        <p class="step-desc">Follow treatment steps, consult AgriDoc AI, and track your farm's recovery</p>
      </div>
    </div>
  </div>
</section>

<!-- DETECTABLE DISEASES -->
<section class="disease-section">
  <div class="disease-inner">
    <div class="sec-head" style="justify-content:center;text-align:center;flex-direction:column;">
      <h2 class="sec-title" style="font-size:1.75rem;">Detectable Plant Diseases</h2>
      <p class="sec-sub" style="margin-top:.4rem;">Our AI recognizes these conditions with high accuracy</p>
    </div>
    <div class="diseases-grid">
      <div class="disease-card" style="border-color:#E74C3C;background:#FEF2F2;">
        <div class="dis-icon">🍂</div>
        <div class="dis-name" style="color:#C0392B;">Anthracnose</div>
        <span class="dis-sev badge-err">High Severity</span>
        <p style="font-size:.75rem;color:#7D2A2A;margin-top:.5rem;">Dark fungal lesions on leaves &amp; stems</p>
      </div>
      <div class="disease-card" style="border-color:#27AE60;background:#F0FFF4;">
        <div class="dis-icon">🌿</div>
        <div class="dis-name" style="color:#1A7A3C;">Healthy</div>
        <span class="dis-sev badge-ok">No Disease</span>
        <p style="font-size:.75rem;color:#1A5C32;margin-top:.5rem;">Vibrant green foliage, normal growth</p>
      </div>
      <div class="disease-card" style="border-color:#E67E22;background:#FFF8F0;">
        <div class="dis-icon">🍃</div>
        <div class="dis-name" style="color:#C0510D;">Leaf Crinkle</div>
        <span class="dis-sev badge-warn">Medium</span>
        <p style="font-size:.75rem;color:#7D3D0D;margin-top:.5rem;">Viral leaf distortion &amp; puckering</p>
      </div>
      <div class="disease-card" style="border-color:#F39C12;background:#FFFBF0;">
        <div class="dis-icon">🌫️</div>
        <div class="dis-name" style="color:#9A6400;">Powdery Mildew</div>
        <span class="dis-sev badge-warn">Medium</span>
        <p style="font-size:.75rem;color:#7D5200;margin-top:.5rem;">White powder coating on leaf surfaces</p>
      </div>
      <div class="disease-card" style="border-color:#F1C40F;background:#FEFFF0;">
        <div class="dis-icon">🟡</div>
        <div class="dis-name" style="color:#9A8000;">Yellow Mosaic</div>
        <span class="dis-sev badge-err">High Severity</span>
        <p style="font-size:.75rem;color:#7D6500;margin-top:.5rem;">Viral yellow mottling patterns</p>
      </div>
    </div>
  </div>
</section>

<!-- CTA -->
<section class="cta-section">
  <h2>Ready to Protect Your Crops?</h2>
  <p>Join thousands of farmers using AI-powered disease detection for better yields</p>
  <div style="display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;">
    <a href="/predict" class="btn btn-white btn-lg">🔬 Start Detection</a>
    <a href="/consultation" class="btn btn-outline-white btn-lg">💬 Ask Expert</a>
  </div>
</section>

<footer style="background:var(--p0);color:rgba(255,255,255,.6);text-align:center;padding:1.5rem;font-size:.82rem;">
  © 2025 PlantDoc AI — Agricultural Disease Detection System &nbsp;|&nbsp; Built with FastAPI + TensorFlow
</footer>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICT PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_prediction_page() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Detect Disease")}<style>
.predict-layout{{display:grid;grid-template-columns:420px 1fr;gap:1.5rem;margin-top:1.5rem;}}
@media(max-width:900px){{.predict-layout{{grid-template-columns:1fr;}}}}

/* upload card */
.upload-zone{{border:2px dashed var(--bd);border-radius:var(--r);padding:2rem;text-align:center;
  background:var(--bg2);cursor:pointer;transition:all .2s;margin-bottom:1rem;}}
.upload-zone:hover,.upload-zone.drag{{border-color:var(--p2);background:#E8F5EE;}}
.upload-zone input{{display:none;}}
.upload-icon{{font-size:2.5rem;margin-bottom:.75rem;}}
.upload-text{{font-size:.88rem;color:var(--tx2);font-weight:500;}}
.upload-hint{{font-size:.75rem;color:var(--tx3);margin-top:.3rem;}}
.preview-img{{width:100%;border-radius:var(--r);margin-bottom:1rem;object-fit:cover;max-height:280px;
  border:1px solid var(--bd);box-shadow:var(--sh);display:none;}}

/* result cards */
.result-placeholder{{text-align:center;padding:4rem 2rem;color:var(--tx3);}}
.result-placeholder .rp-icon{{font-size:4rem;margin-bottom:1rem;opacity:.4;}}

.disease-result-header{{display:flex;align-items:flex-start;gap:1rem;margin-bottom:1.25rem;}}
.disease-badge-big{{width:56px;height:56px;border-radius:14px;display:flex;align-items:center;
  justify-content:center;font-size:1.6rem;flex-shrink:0;}}
.disease-name-big{{font-size:1.35rem;font-weight:800;color:var(--tx);}}
.disease-desc{{font-size:.85rem;color:var(--tx2);margin-top:.25rem;}}

.conf-row{{display:flex;align-items:center;gap.5rem;margin-bottom:.25rem;}}
.conf-label{{font-size:.8rem;font-weight:600;color:var(--tx2);min-width:90px;}}
.conf-value{{font-size:.88rem;font-weight:700;color:var(--p1);}}

.info-pill-row{{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0;}}

.detail-block{{background:var(--bg2);border-radius:10px;padding:1rem 1.1rem;margin-bottom:.75rem;}}
.detail-block-title{{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
  color:var(--tx3);margin-bottom:.5rem;display:flex;align-items:center;gap:.4rem;}}
.detail-list{{list-style:none;}}
.detail-list li{{font-size:.85rem;color:var(--tx2);padding:.2rem 0;border-bottom:1px solid var(--bd2);}}
.detail-list li:last-child{{border-bottom:none;}}
.detail-list li::before{{content:"→ ";color:var(--p2);font-weight:700;}}

.adv-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;margin-top:.5rem;}}
.adv-cell{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:.6rem;text-align:center;}}
.adv-cell-name{{font-size:.7rem;font-weight:700;color:var(--tx3);text-transform:uppercase;margin-bottom:.25rem;}}
.adv-cell-val{{font-size:.82rem;font-weight:700;color:var(--p1);}}

.smart-box{{background:linear-gradient(135deg,var(--p1),var(--p2));color:#fff;border-radius:var(--r);
  padding:1rem 1.1rem;margin-bottom:.75rem;}}
.smart-box .sb-label{{font-size:.75rem;opacity:.8;text-transform:uppercase;letter-spacing:.04em;}}
.smart-box .sb-value{{font-size:.9rem;font-weight:700;margin-top:.15rem;}}

.save-diary-btn{{margin-top:1rem;}}
</style>
</head>
<body>
{build_nav("/predict")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>🔬 Plant Disease Detection</h1>
    <p>Upload a leaf photo for instant AI-powered diagnosis with treatment recommendations</p>
    <div class="hero-badges">
      <span class="hero-badge">🛡️ Adversarial Defense</span>
      <span class="hero-badge">🌦️ Weather Correction</span>
      <span class="hero-badge">⚡ Instant Results</span>
    </div>
  </div>
</div>

<div class="pda-container">
  <div class="predict-layout">

    <!-- LEFT: Upload Panel -->
    <div>
      <div class="pda-card">
        <div class="pda-card-header">📸 Upload Plant Image</div>
        <div class="pda-card-body">
          <label class="upload-zone" id="dropZone">
            <input type="file" id="imageFile" accept="image/*">
            <div class="upload-icon">🌿</div>
            <div class="upload-text">Click to select or drag &amp; drop a photo</div>
            <div class="upload-hint">Supports JPG, PNG, WEBP  • Max 10 MB</div>
          </label>
          <img id="previewImg" class="preview-img" alt="Preview">

          <div class="form-group">
            <label class="form-label">Device / Location ID</label>
            <input type="text" id="deviceId" class="form-control" placeholder="e.g. Field-A, Raspi-1">
          </div>

          <button id="predictBtn" class="btn btn-primary btn-lg btn-block" onclick="runDetection()">
            🔍 Analyze Plant Health
          </button>

          <div id="uploadStatus" style="margin-top:.75rem;"></div>
        </div>
      </div>

      <!-- Tips -->
      <div class="pda-card" style="margin-top:1rem;">
        <div class="pda-card-header">💡 Tips for Best Results</div>
        <div class="pda-card-body" style="font-size:.83rem;color:var(--tx2);">
          <p style="margin-bottom:.5rem;">✅ Good lighting — avoid dark or blurry shots</p>
          <p style="margin-bottom:.5rem;">✅ Fill frame with the leaf — close-up is better</p>
          <p style="margin-bottom:.5rem;">✅ Show affected area clearly</p>
          <p style="margin-bottom:.5rem;">✅ Avoid photos with shadows or glare</p>
          <p>✅ One leaf per photo for best accuracy</p>
        </div>
      </div>
    </div>

    <!-- RIGHT: Results Panel -->
    <div>
      <div class="pda-card" id="resultsCard">
        <div class="pda-card-header">📊 Analysis Results</div>
        <div class="pda-card-body" id="resultsBody">
          <div class="result-placeholder">
            <div class="rp-icon">🌱</div>
            <h3 style="font-size:1rem;font-weight:700;color:var(--tx2);margin-bottom:.4rem;">Waiting for Image</h3>
            <p style="font-size:.85rem;">Upload a plant leaf photo to get instant disease detection and treatment recommendations.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('imageFile');
const previewImg = document.getElementById('previewImg');

fileInput.addEventListener('change', e => {{
  const file = e.target.files[0];
  if(file) showPreview(file);
}});

dropZone.addEventListener('dragover', e => {{ e.preventDefault(); dropZone.classList.add('drag'); }});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {{
  e.preventDefault(); dropZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if(file) {{ fileInput.files = e.dataTransfer.files; showPreview(file); }}
}});

function showPreview(file) {{
  const reader = new FileReader();
  reader.onload = ev => {{ previewImg.src = ev.target.result; previewImg.style.display='block'; }};
  reader.readAsDataURL(file);
}}

async function runDetection() {{
  const file = fileInput.files[0];
  if(!file) {{ showToast('⚠️ Please select an image first'); return; }}
  const btn = document.getElementById('predictBtn');
  const status = document.getElementById('uploadStatus');
  const body = document.getElementById('resultsBody');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing...';
  status.innerHTML = '';
  body.innerHTML = `<div style="text-align:center;padding:3rem;color:var(--tx3);">
    <span class="spinner" style="width:32px;height:32px;border-width:3px;"></span>
    <p style="margin-top:1rem;font-size:.9rem;font-weight:600;">Processing image with AI + weather correction...</p></div>`;

  const form = new FormData();
  form.append('file', file);
  const deviceId = document.getElementById('deviceId').value;
  if(deviceId) form.append('device_id', deviceId);

  try {{
    const res = await fetch('/detect', {{ method:'POST', body:form }});
    const data = await res.json();
    if(data.status === 'success') renderResults(data);
    else {{
      const msg = data.plant_detected === false
        ? `<div class="alert alert-warn">🌿 <div><strong>No Plant Detected</strong><br>${{data.detail}}<br><small style="opacity:.8">${{data.message}}</small></div></div>`
        : `<div class="alert alert-err">❌ ${{data.detail}}</div>`;
      body.innerHTML = msg;
    }}
  }} catch(e) {{
    body.innerHTML = `<div class="alert alert-err">❌ Network error: ${{e.message}}</div>`;
  }} finally {{
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyze Plant Health';
  }}
}}

function renderResults(data) {{
  const p = data.prediction;
  const sa = p.smart_analysis || {{}};
  const wc = p.weather_conditions || {{}};
  const aa = p.adversarial_analysis || {{}};
  const at = p.adversarial_tests || {{}};
  const conf = (p.confidence * 100).toFixed(1);
  const sevClass = {{ High:'badge-err', Medium:'badge-warn', None:'badge-ok' }}[p.severity] || 'badge-gray';
  const disIcon = {{
    'Anthracnose':'🍂','Healthy':'🌿','Leaf Crinkle':'🍃',
    'Powdery Mildew':'🌫️','Yellow Mosaic':'🟡'
  }}[p.disease_name] || '🌱';

  let html = '';

  // Header
  html += `<div class="disease-result-header">
    <div class="disease-badge-big" style="background:var(--bg2);">${{disIcon}}</div>
    <div>
      <div class="disease-name-big">${{p.disease_name}}</div>
      <div class="disease-desc">${{p.description}}</div>
      <div class="info-pill-row" style="margin-top:.5rem;">
        <span class="badge ${{sevClass}}">${{p.severity}} Severity</span>
        <span class="badge badge-green">${{conf}}% Confidence</span>
      </div>
    </div>
  </div>`;

  // Confidence bar
  html += `<div style="margin-bottom:1rem;">
    <div style="display:flex;justify-content:space-between;font-size:.8rem;color:var(--tx2);margin-bottom:.25rem;">
      <span>Detection Confidence</span><span style="font-weight:700;color:var(--p1);">${{conf}}%</span>
    </div>
    <div class="pbar-wrap"><div class="pbar-fill" style="width:${{conf}}%"></div></div>
    <div style="font-size:.75rem;color:var(--tx3);margin-top:.25rem;">${{sa.interpretation || ''}}</div>
  </div>`;

  // Smart Analysis
  if(sa.secondary) {{
    html += `<div class="smart-box">
      <div class="sb-label">Smart Analysis</div>
      <div class="sb-value">Primary: ${{sa.primary}} (${{(sa.primary_confidence*100).toFixed(1)}}%)</div>
      <div class="sb-value" style="opacity:.8;">Secondary: ${{sa.secondary}} (${{(sa.secondary_confidence*100).toFixed(1)}}%)</div>
    </div>`;
  }}

  // Weather & Adversarial alerts
  if(wc.foggy_detected || wc.dark_conditions || wc.low_contrast) {{
    html += `<div class="alert alert-warn">🌦️ Weather conditions detected — image automatically enhanced for better accuracy.</div>`;
  }}
  if(aa.attack_detected) {{
    html += `<div class="alert alert-err">🛡️ Adversarial patterns detected — defense mechanisms applied.</div>`;
  }}
  if(p.overconfidence_warning) {{
    html += `<div class="alert alert-warn">⚠️ ${{p.overconfidence_warning}}</div>`;
  }}
  // ── YOUR NEW FEATURE ──
if(p.warning){{
  html += `<div class="alert alert-warn">${{p.warning}}</div>`;
}}

html += `
  <div class="detail-block">
    <div class="detail-block-title">🧠 Why this prediction</div>
    <p style="font-size:.85rem;color:var(--tx2);">${{p.explanation}}</p>
  </div>
`;
  

  // Symptoms
  if(p.symptoms?.length) {{
    html += `<div class="detail-block">
      <div class="detail-block-title">🔍 Symptoms</div>
      <ul class="detail-list">${{p.symptoms.map(s=>`<li>${{s}}</li>`).join('')}}</ul>
    </div>`;
  }}

  // Treatment & Prevention
  html += `<div class="detail-block">
    <div class="detail-block-title">💊 Treatment</div>
    <p style="font-size:.85rem;color:var(--tx2);">${{p.treatment}}</p>
  </div>`;
  html += `<div class="detail-block">
    <div class="detail-block-title">🛡️ Prevention</div>
    <p style="font-size:.85rem;color:var(--tx2);">${{p.prevention}}</p>
  </div>`;

  // Organic solutions
  if(p.organic_solutions?.length) {{
    html += `<div class="detail-block">
      <div class="detail-block-title">🌿 Organic Solutions</div>
      <ul class="detail-list">${{p.organic_solutions.map(s=>`<li>${{s}}</li>`).join('')}}</ul>
    </div>`;
  }}

  // Adversarial test results
  if(at.fgsm_prediction || at.pgd_prediction || at.cw_prediction) {{
    html += `<div class="detail-block">
      <div class="detail-block-title">🛡️ Adversarial Robustness Tests</div>
      <div class="adv-grid">
        <div class="adv-cell"><div class="adv-cell-name">FGSM</div><div class="adv-cell-val">${{at.fgsm_prediction||'—'}}</div></div>
        <div class="adv-cell"><div class="adv-cell-name">PGD</div><div class="adv-cell-val">${{at.pgd_prediction||'—'}}</div></div>
        <div class="adv-cell"><div class="adv-cell-name">C&W</div><div class="adv-cell-val">${{at.cw_prediction||'—'}}</div></div>
      </div>
    </div>`;
  }}

  // Save to diary
  html += `<button class="btn btn-outline btn-block save-diary-btn" onclick="saveToDiary(${{JSON.stringify(p).replace(/'/g,"\\'")}})">
    📔 Save to Farm Diary
  </button>`;

  document.getElementById('resultsBody').innerHTML = html;
}}

async function saveToDiary(pred) {{
  try {{
    const res = await fetch('/diary/save', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        type:'detection', title:`Disease Detected: ${{pred.disease_name}}`,
        content:{{disease_name:pred.disease_name,confidence:pred.confidence,severity:pred.severity}},
        tags:['detection',pred.disease_name.toLowerCase().replace(' ','_')], crop:'plant', language:'english'
      }})
    }});
    const d = await res.json();
    showToast(d.status==='success' ? '📔 Saved to Farm Diary!' : '⚠️ Could not save: '+d.detail);
  }} catch(e) {{ showToast('Network error while saving'); }}
}}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  RECORDS PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_records_page() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Detection Records")}<style>
.filter-bar{{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin-bottom:1.5rem;
  background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:.9rem 1.1rem;}}
.filter-bar .form-control{{max-width:200px;}}

.records-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1.25rem;}}

.rec-card{{background:var(--card);border-radius:var(--r);border:1px solid var(--bd);
  overflow:hidden;box-shadow:var(--sh);transition:all .2s;}}
.rec-card:hover{{transform:translateY(-3px);box-shadow:var(--shm);}}
.rec-img{{width:100%;height:180px;object-fit:cover;background:var(--bg2);display:block;}}
.rec-body{{padding:1rem 1.1rem;}}
.rec-header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.6rem;}}
.rec-disease{{font-size:.98rem;font-weight:800;color:var(--tx);}}
.rec-conf{{font-size:.75rem;font-weight:700;padding:.18rem .55rem;border-radius:20px;
  background:var(--p1);color:#fff;flex-shrink:0;}}
.rec-meta{{font-size:.75rem;color:var(--tx3);margin-bottom:.65rem;}}
.rec-treatment{{font-size:.8rem;color:var(--tx2);line-height:1.4;margin-bottom:.75rem;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}}
.rec-footer{{display:flex;align-items:center;justify-content:space-between;
  padding:.7rem 1.1rem;background:var(--bg2);border-top:1px solid var(--bd2);}}
.rec-flags{{display:flex;gap:.4rem;flex-wrap:wrap;}}

.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}}
@media(max-width:700px){{.stats-row{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
{build_nav("/records-page")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>📊 Detection Records</h1>
    <p>Complete history of all plant disease detections with analysis and treatment logs</p>
  </div>
</div>

<div class="pda-container">

  <!-- Stats row -->
  <div class="stats-row" id="statsRow">
    <div class="stat-card"><div class="stat-icon">📋</div><div class="stat-num" id="sTotal">—</div><div class="stat-lbl">Total Records</div></div>
    <div class="stat-card"><div class="stat-icon">🦠</div><div class="stat-num" id="sDiseased">—</div><div class="stat-lbl">Diseases Found</div></div>
    <div class="stat-card"><div class="stat-icon">🌿</div><div class="stat-num" id="sHealthy">—</div><div class="stat-lbl">Healthy Plants</div></div>
    <div class="stat-card"><div class="stat-icon">🛡️</div><div class="stat-num" id="sDefended">—</div><div class="stat-lbl">Attacks Defended</div></div>
  </div>

  <!-- Filter bar -->
  <div class="filter-bar">
    <input type="text" id="searchInput" class="form-control" placeholder="🔍 Search records..." oninput="filterRecords()">
    <select id="severityFilter" class="form-control" onchange="filterRecords()">
      <option value="">All Severities</option>
      <option value="High">High Severity</option>
      <option value="Medium">Medium Severity</option>
      <option value="None">Healthy</option>
    </select>
    <select id="sortFilter" class="form-control" onchange="filterRecords()">
      <option value="newest">Newest First</option>
      <option value="oldest">Oldest First</option>
      <option value="confidence">By Confidence</option>
    </select>
    <button class="btn btn-ghost btn-sm" onclick="loadRecords()">🔄 Refresh</button>
    <span id="recCount" class="badge badge-gray" style="margin-left:auto;"></span>
  </div>

  <!-- Grid -->
  <div id="recordsGrid">
    <div class="empty-state">
      <div class="es-icon"><span class="spinner" style="width:40px;height:40px;border-width:4px;"></span></div>
      <h3>Loading records...</h3>
    </div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
let allRecords = [];

async function loadRecords() {{
  try {{
    const res = await fetch('/records?limit=1000');
    const data = await res.json();
    if(data.status === 'success') {{
      allRecords = data.records;
      updateStats(allRecords);
      filterRecords();
    }}
  }} catch(e) {{
    document.getElementById('recordsGrid').innerHTML =
      '<div class="alert alert-err" style="margin-top:2rem;">❌ Failed to load records. Please check connection.</div>';
  }}
}}

function updateStats(records) {{
  document.getElementById('sTotal').textContent = records.length;
  document.getElementById('sDiseased').textContent = records.filter(r=>r.prediction?.severity !== 'None').length;
  document.getElementById('sHealthy').textContent = records.filter(r=>r.prediction?.severity === 'None').length;
  document.getElementById('sDefended').textContent = records.filter(r=>r.processing_analysis?.is_adversarial).length;
}}

function filterRecords() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  const sev = document.getElementById('severityFilter').value;
  const sort = document.getElementById('sortFilter').value;
  let filtered = allRecords.filter(r => {{
    const disease = (r.prediction?.disease_name||'').toLowerCase();
    const device = (r.device_id||'').toLowerCase();
    if(q && !disease.includes(q) && !device.includes(q)) return false;
    if(sev && r.prediction?.severity !== sev) return false;
    return true;
  }});
  if(sort === 'oldest') filtered.sort((a,b) => a.timestamp - b.timestamp);
  else if(sort === 'confidence') filtered.sort((a,b) => (b.prediction?.confidence||0) - (a.prediction?.confidence||0));
  else filtered.sort((a,b) => b.timestamp - a.timestamp);
  document.getElementById('recCount').textContent = `${{filtered.length}} record${{filtered.length!==1?'s':''}}`;
  renderRecords(filtered);
}}

function renderRecords(records) {{
  const grid = document.getElementById('recordsGrid');
  if(!records.length) {{
    grid.innerHTML = `<div class="empty-state">
      <div class="es-icon">📭</div>
      <h3>No Records Found</h3>
      <p>Try changing filters or <a href="/predict" style="color:var(--p1);">detect a new disease</a>.</p>
    </div>`;
    return;
  }}
  grid.innerHTML = `<div class="records-grid">${{records.map(rec => {{
    const pred = rec.prediction || {{}};
    const date = new Date(rec.timestamp * 1000).toLocaleString('en-IN', {{
      day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'
    }});
    const sevClass = {{High:'badge-err',Medium:'badge-warn',None:'badge-ok'}}[pred.severity] || 'badge-gray';
    const conf = ((pred.confidence||0)*100).toFixed(1);
    const flags = [];
    if(rec.processing_analysis?.is_adversarial) flags.push('<span class="badge badge-err">🛡️ Defended</span>');
    if(rec.processing_analysis?.is_foggy) flags.push('<span class="badge badge-warn">🌫️ Enhanced</span>');
    if(rec.processing_analysis?.is_dark) flags.push('<span class="badge badge-warn">🌙 Brightened</span>');
    return `<div class="rec-card">
      <img src="/image/${{rec.enhanced_image||rec.original_image}}" class="rec-img"
           alt="${{pred.disease_name}}" onerror="this.style.background='var(--bg2)'">
      <div class="rec-body">
        <div class="rec-header">
          <div class="rec-disease">${{pred.disease_name||'Unknown'}}</div>
          <span class="rec-conf">${{conf}}%</span>
        </div>
        <div class="rec-meta">
          <span class="badge ${{sevClass}}">${{pred.severity}} Severity</span>
          &nbsp; 📅 ${{date}}
        </div>
        <div class="rec-meta">📡 Device: ${{rec.device_id||'—'}}</div>
        <p class="rec-treatment">${{pred.treatment||'—'}}</p>
      </div>
      <div class="rec-footer">
        <div class="rec-flags">${{flags.join('')}}</div>
        <button class="btn btn-danger btn-xs" onclick="deleteRecord('${{rec._id}}')">🗑️ Delete</button>
      </div>
    </div>`;
  }}).join('')}}</div>`;
}}

async function deleteRecord(id) {{
  if(!confirm('Delete this record permanently?')) return;
  try {{
    const res = await fetch(`/records/${{id}}`, {{method:'DELETE'}});
    const data = await res.json();
    if(data.status === 'success') {{ showToast('✅ Record deleted'); loadRecords(); }}
    else showToast('⚠️ ' + data.detail);
  }} catch(e) {{ showToast('❌ Network error'); }}
}}

loadRecords();
setInterval(loadRecords, 60000);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBCAM PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_webcam_page() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>{shared_head("Live Camera")}<style>
.webcam-layout{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:1.5rem;}}
@media(max-width:850px){{.webcam-layout{{grid-template-columns:1fr;}}}}

.cam-placeholder{{background:var(--bg2);border:2px dashed var(--bd);border-radius:var(--r);
  padding:3rem;text-align:center;color:var(--tx3);}}
#video{{width:100%;border-radius:var(--r);display:none;box-shadow:var(--shm);}}
#canvas{{display:none;}}

.cam-controls{{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1rem;}}

.auto-badge{{display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;font-weight:600;
  padding:.3rem .7rem;border-radius:20px;background:#E8F5EE;color:var(--p1);border:1px solid var(--p3);}}
.auto-badge.live{{background:var(--p1);color:#fff;animation:pulse 1.5s ease-in-out infinite;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.7}}}}

.live-result-card{{min-height:300px;}}
.lr-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;}}
.lr-status-dot{{width:10px;height:10px;border-radius:50%;background:#ccc;}}
.lr-status-dot.live{{background:#27AE60;animation:pulse 1.5s infinite;}}

.lrd-name{{font-size:1.25rem;font-weight:800;color:var(--tx);}}
.lrd-conf{{font-size:.9rem;color:var(--p1);font-weight:600;}}
.lrd-time{{font-size:.75rem;color:var(--tx3);}}
.lrd-detail{{background:var(--bg2);border-radius:8px;padding:.85rem 1rem;font-size:.85rem;color:var(--tx2);margin-top:.75rem;}}
</style>
</head>
<body>
{build_nav("/webcam")}

<div class="pda-hero">
  <div class="pda-hero-inner">
    <h1>📹 Live Camera Detection</h1>
    <p>Real-time plant disease monitoring — auto-analyzes every 5 seconds</p>
    <div class="hero-badges">
      <span class="hero-badge">🔄 Auto-Detect Every 5s</span>
      <span class="hero-badge">🛡️ Defense Active</span>
      <span class="hero-badge">📱 Mobile Friendly</span>
    </div>
  </div>
</div>

<div class="pda-container">
  <div class="webcam-layout">

    <!-- Camera feed -->
    <div class="pda-card">
      <div class="pda-card-header">📷 Camera Feed</div>
      <div class="pda-card-body">
        <div id="camPlaceholder" class="cam-placeholder">
          <div style="font-size:3rem;margin-bottom:.75rem;">📷</div>
          <p style="font-weight:600;font-size:.9rem;">Camera not started</p>
          <p style="font-size:.8rem;margin-top:.3rem;">Click Start Camera below to begin live detection</p>
        </div>
        <video id="video" autoplay muted playsinline></video>
        <canvas id="canvas"></canvas>

        <div class="cam-controls">
          <button id="startBtn" class="btn btn-primary" onclick="startCamera()">▶️ Start Camera</button>
          <button id="stopBtn" class="btn btn-ghost" onclick="stopCamera()" style="display:none;">⏹ Stop</button>
          <button id="captureBtn" class="btn btn-accent" onclick="captureNow()" style="display:none;">📸 Analyze Now</button>
        </div>

        <div id="autoInfo" style="display:none;margin-top:.75rem;">
          <span class="auto-badge live" id="autoStatus">🔄 Auto-detecting every 5 seconds</span>
        </div>
      </div>
    </div>

    <!-- Live results -->
    <div class="pda-card live-result-card">
      <div class="pda-card-header">
        <span>🔬 Live Analysis</span>
        <div class="lr-status-dot" id="statusDot"></div>
      </div>
      <div class="pda-card-body" id="liveBody">
        <div class="cam-placeholder" style="min-height:260px;padding:2rem;border:none;">
          <div style="font-size:2.5rem;margin-bottom:.75rem;">🌱</div>
          <p style="font-weight:600;">Waiting for camera feed</p>
          <p style="font-size:.8rem;margin-top:.3rem;">Start the camera to see live results here</p>
        </div>
      </div>
    </div>
  </div>
</div>

{TOAST_SCRIPT}
<script>
let video, canvas, ctx, stream = null, isAnalyzing = false, intervalId = null;

document.addEventListener('DOMContentLoaded', () => {{
  video = document.getElementById('video');
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
}});

async function startCamera() {{
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{width:{{ideal:640}}, height:{{ideal:480}}, facingMode:'environment'}}
    }});
    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('camPlaceholder').style.display = 'none';
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'inline-flex';
    document.getElementById('captureBtn').style.display = 'inline-flex';
    document.getElementById('autoInfo').style.display = 'block';
    document.getElementById('statusDot').classList.add('live');
    video.addEventListener('loadedmetadata', () => {{
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }});
    intervalId = setInterval(autoDetect, 5000);
  }} catch(e) {{
    showToast('❌ Camera error: ' + e.message);
  }}
}}

function stopCamera() {{
  if(stream) stream.getTracks().forEach(t => t.stop()); stream = null;
  if(intervalId) clearInterval(intervalId); intervalId = null;
  video.style.display = 'none';
  document.getElementById('camPlaceholder').style.display = 'block';
  document.getElementById('camPlaceholder').innerHTML = '<div style="font-size:3rem;margin-bottom:.75rem;">⏹</div><p style="font-weight:600;">Camera stopped</p>';
  document.getElementById('startBtn').style.display = 'inline-flex';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('captureBtn').style.display = 'none';
  document.getElementById('autoInfo').style.display = 'none';
  document.getElementById('statusDot').classList.remove('live');
}}

function captureNow() {{
  if(!stream || isAnalyzing) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => analyzeBlob(blob, false), 'image/jpeg', 0.85);
}}

function autoDetect() {{
  if(!stream || isAnalyzing) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => analyzeBlob(blob, true), 'image/jpeg', 0.65);
}}

async function analyzeBlob(blob, isAuto) {{
  if(isAnalyzing) return;
  isAnalyzing = true;
  const body = document.getElementById('liveBody');
  if(!isAuto) body.innerHTML = '<div style="text-align:center;padding:2rem;"><span class="spinner" style="width:32px;height:32px;border-width:3px;"></span><p style="margin-top:.75rem;font-size:.85rem;color:var(--tx2);">Analyzing frame...</p></div>';

  try {{
    const form = new FormData();
    form.append('file', blob, 'webcam.jpg');
    form.append('device_id', 'webcam_' + Date.now());
    const res = await fetch('/detect', {{method:'POST', body:form}});
    const data = await res.json();
    if(data.status === 'success') renderLive(data, isAuto);
    else if(!isAuto) {{
      body.innerHTML = data.plant_detected === false
        ? `<div class="alert alert-warn">🌿 <div><strong>No Plant Detected</strong><br>${{data.detail}}</div></div>`
        : `<div class="alert alert-err">❌ ${{data.detail}}</div>`;
    }}
  }} catch(e) {{
    if(!isAuto) body.innerHTML = `<div class="alert alert-err">❌ Network error: ${{e.message}}</div>`;
  }} finally {{ isAnalyzing = false; }}
}}

function renderLive(data, isAuto) {{
  const p = data.prediction;
  const conf = (p.confidence*100).toFixed(1);
  const time = new Date().toLocaleTimeString();
  const sevClass = {{High:'badge-err',Medium:'badge-warn',None:'badge-ok'}}[p.severity]||'badge-gray';
  const wc = p.weather_conditions || {{}};
  const aa = p.adversarial_analysis || {{}};

  let html = `<div>
    <div class="lrd-name">${{p.disease_name}}</div>
    <div class="lrd-conf">${{conf}}% Confidence</div>
    <div class="lrd-time">${{time}} ${{isAuto?'· Auto':'· Manual'}}</div>
    <div class="info-pill-row" style="margin:.6rem 0;">
      <span class="badge ${{sevClass}}">${{p.severity}} Severity</span>
    </div>
    <div class="pbar-wrap"><div class="pbar-fill" style="width:${{conf}}%"></div></div>
    <div class="lrd-detail">
      <strong>Description:</strong> ${{p.description}}<br>
      <strong>Treatment:</strong> ${{(p.treatment||'').substring(0,120)}}${{p.treatment?.length>120?'...':''}}
    </div>`;

  if(wc.foggy_detected||wc.dark_conditions)
    html += `<div class="alert alert-warn" style="margin-top:.5rem;">🌦️ Weather enhancement applied</div>`;
  if(aa.attack_detected)
    html += `<div class="alert alert-err" style="margin-top:.5rem;">🛡️ Defense mechanisms activated</div>`;

  html += `</div>`;
  document.getElementById('liveBody').innerHTML = html;
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)