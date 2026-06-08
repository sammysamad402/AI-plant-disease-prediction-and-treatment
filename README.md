---
title: PlantDoc AI
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🌿 PlantDoc AI — Robust Defense System Against Adversarial Attacks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://plantdoc-ai-u883.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)

**AI-powered plant disease detection with adversarial attack defense, JWT authentication, multilingual expert chat, and full farm management — live on Render.**

🔗 **Live Demo:** [plantdoc-ai-u883.onrender.com](https://plantdoc-ai-u883.onrender.com)

[Features](#-features) • [Architecture](#️-system-architecture) • [Auth](#-authentication) • [Installation](#️-installation) • [Deployment](#-deployment-on-render) • [Model](#-model--weights) • [Results](#-results--performance)

</div>

---

## 📌 About The Project

PlantDoc AI is a full-stack agricultural intelligence platform that detects plant diseases from leaf images using a custom-trained CNN, while actively defending against adversarial attacks — invisible pixel manipulations that fool standard AI models.

> **Problem:** Existing plant disease AI systems achieve high accuracy in lab settings but are completely undefended against adversarial noise. A perturbation invisible to the human eye can drop model accuracy from 91% to 11%.

> **Our Solution:** A 5-layer defense pipeline implementing FGSM, PGD, and C&W attack simulation with Gaussian blur and median filter defenses — recovering up to 79% accuracy under attack. Combined with full JWT-based user authentication so every farmer's data stays private.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **JWT Authentication** | Full register/login system — each user's detections, diary, and consultations are private |
| 🔬 **Disease Detection** | CNN model with 91.4% accuracy across 5 disease classes |
| 🛡️ **Adversarial Defense** | FGSM, PGD & C&W attack simulation + 5-layer defense pipeline |
| 🌦️ **Weather Enhancement** | Auto-corrects foggy, dark, and low-contrast field images |
| 📹 **Live Webcam** | Real-time detection every 5 seconds with auto-defense |
| 💬 **AgriDoc AI Chat** | Multilingual expert AI in 9 Indian languages with voice I/O |
| 📔 **Farm Diary** | Personal detection & consultation history with tags and filters |
| 🗓️ **Crop Calendar** | Month-by-month farming guide for 5 major Indian crops |
| 🏪 **Agri Shops Map** | GPS-based map of nearby agricultural stores and KVK centers |

---

## 🦠 Detectable Diseases

| Disease | Severity | Description |
|---|---|---|
| 🍂 Anthracnose | 🔴 High | Dark fungal lesions on leaves and stems |
| 🌿 Healthy | ✅ None | Vibrant green foliage, no disease signs |
| 🍃 Leaf Crinkle | 🟡 Medium | Viral leaf distortion and puckering |
| 🌫️ Powdery Mildew | 🟡 Medium | White powder coating on leaf surfaces |
| 🟡 Yellow Mosaic | 🔴 High | Viral yellow mottling patterns |

---

## 🏗️ System Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────┐
│              5-LAYER DEFENSE PIPELINE            │
│                                                  │
│  Layer 1 → Plant Verification  (HSV Analysis)   │
│  Layer 2 → Weather Enhancement (PIL Filters)    │
│  Layer 3 → Adversarial Detection (Sobel Grad.)  │
│  Layer 4 → Signal Smoothing (Gaussian + Median) │
│  Layer 5 → Confidence Gating  (≥60% threshold)  │
└─────────────────────────────────────────────────┘
    │
    ▼
BPLD CNN Model (224×224 input → 5-class Softmax)
    │
    ▼
Disease Name + Confidence + Severity + Treatment
```

### CNN Architecture

```
Input (224×224×3)
→ Conv2D(32, 3×3, ReLU) → MaxPool
→ Conv2D(64, 3×3, ReLU) → MaxPool
→ Conv2D(128, 3×3, ReLU) → MaxPool
→ Flatten
→ Dense(128, ReLU) → Dropout(0.5)
→ Dense(5, Softmax)
```

### Adversarial Attacks Tested

| Attack | Method | Accuracy Without Defense | Accuracy With Defense |
|---|---|---|---|
| FGSM | Fast Gradient Sign (ε=0.03) | 28% | **79.2%** |
| PGD | Projected Gradient Descent (5 iters) | 11% | **71.0%** |
| C&W | Carlini & Wagner (L2 optimize) | 19% | **68.3%** |

---

## 🔐 Authentication

PlantDoc AI uses **JWT (JSON Web Token)** based authentication — every user's data is fully private and scoped to their account.

### How it works

- **Register** with name, email, and password — stored securely with bcrypt hashing
- **Login** returns a JWT token (7-day expiry) stored in `localStorage`
- Every API request sends the token in the `Authorization: Bearer <token>` header
- All detections, diary entries, and consultations are tagged with `user_id`
- Records page only returns data belonging to the logged-in user

### Auth Endpoints

```
POST  /auth/register   # Create new account
POST  /auth/login      # Login, returns JWT token
GET   /login           # Login page
GET   /register        # Register page
```

### Environment variable required

```env
JWT_SECRET_KEY=your-long-random-secret-key
JWT_EXPIRE_MINUTES=10080   # 7 days (default)
```

---

## 🛠️ Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework
- [TensorFlow / Keras 2.13](https://tensorflow.org) — CNN model inference
- [OpenCV](https://opencv.org) — Image processing & adversarial detection
- [Pillow (PIL)](https://pillow.readthedocs.io) — Weather enhancement filters
- [MongoDB Atlas](https://mongodb.com) — Database for users, detections & diary
- [python-jose](https://github.com/mpdavis/python-jose) — JWT token handling
- [passlib + bcrypt](https://passlib.readthedocs.io) — Password hashing
- [h5py](https://www.h5py.org) — Version-safe model weight storage
- [gdown](https://github.com/wkentaro/gdown) — Google Drive model download

**Frontend**
- HTML5, CSS3, JavaScript — Unified web UI served directly from FastAPI
- Inter font — consistent typography across all pages

**AI Services**
- [OpenAI GPT-4o-mini](https://openai.com) — AgriDoc multilingual expert chat
- Web Speech API — voice input and text-to-speech output

**DevOps**
- [Render](https://render.com) — Cloud deployment
- [Google Drive + gdown](https://github.com/wkentaro/gdown) — Model weight hosting & auto-download
- [Uvicorn](https://uvicorn.org) — ASGI production server
- python-dotenv — environment variable management

---

## 📁 Project Structure

```
AI-plant-disease-prediction-and-treatment/
│
├── main.py              # FastAPI app — detection, records, all page routes
├── auth.py              # JWT authentication — register, login, token verification
├── consultation.py      # AgriDoc AI chat — multilingual expert consultation
├── diary.py             # Farm diary + agri shops page
├── crop_calendar.py     # Month-by-month crop farming guide
├── visualization.py     # Data visualization & analytics
├── shared_ui.py         # Unified design system — CSS, navbar, auth helpers
│
├── training/            # Training notebooks (empty — local only)
│
├── requirements.txt     # Python dependencies
├── .python-version      # Python version pin (3.11)
├── .gitignore
├── .env                 # Environment variables (not committed)
└── README.md
```

> ⚠️ The model file (`weights_raw.h5`) is **not included** in the repo due to size. It is auto-downloaded from Google Drive at startup. See the [Model & Weights](#-model--weights) section below.

---

## ⚙️ Installation

### Prerequisites
- Python 3.11
- MongoDB (local or [Atlas](https://mongodb.com/atlas))
- OpenAI API key (for AgriDoc chat)

### 1. Clone the repository

```bash
git clone https://github.com/sammysamad402/AI-plant-disease-prediction-and-treatment.git
cd AI-plant-disease-prediction-and-treatment
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
MONGO_URI=mongodb://localhost:27017/
MODEL_PATH=weights_raw.h5
MODEL_DRIVE_ID=your_google_drive_file_id
UPLOAD_DIR=uploads
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET_KEY=your-long-random-secret-key-here
JWT_EXPIRE_MINUTES=10080
```

### 5. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000** — you'll be redirected to login/register first.

---

## 🚀 Usage

### Web Application

| Page | URL | Auth Required |
|---|---|---|
| Home | `/` | ✅ |
| Detect | `/predict` | ✅ |
| Live Cam | `/webcam` | ✅ |
| Records | `/records-page` | ✅ (own records only) |
| Consult | `/consultation` | ✅ |
| Diary | `/diary` | ✅ |
| Calendar | `/crop-calendar` | ✅ |
| Agri Shops | `/agri-shops` | ✅ |
| Login | `/login` | ❌ |
| Register | `/register` | ❌ |

### API Endpoints

```
POST   /detect              # Upload image for disease detection
GET    /records             # Fetch current user's detection history
DELETE /records/{id}        # Delete own record
GET    /image/{filename}    # Serve uploaded image
POST   /ask-expert          # Send message to AgriDoc AI
POST   /diary/save          # Save diary entry
GET    /diary/entries       # Get current user's diary entries
GET    /crop-calendar       # Crop calendar page
POST   /auth/register       # Register new user
POST   /auth/login          # Login and receive JWT token
```

---

## ☁️ Deployment on Render

### Environment Variables (set in Render dashboard)

| Variable | Value |
|---|---|
| `MONGO_URI` | Your MongoDB Atlas connection string |
| `MODEL_DRIVE_ID` | Google Drive file ID of `weights_raw.h5` |
| `MODEL_PATH` | `weights_raw.h5` |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `JWT_SECRET_KEY` | A long random secret string |
| `JWT_EXPIRE_MINUTES` | `10080` (7 days) |

### How auto-download works

On startup, `main.py` checks if `weights_raw.h5` exists. If not, it downloads from Google Drive:

```python
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={drive_id}", MODEL_PATH, quiet=False)
```

---

## 🧠 Model & Weights

### The Cross-Version Keras Problem

The model was trained locally on a newer Keras version but Render runs TensorFlow 2.13 / Keras 2.13. Direct `.h5` loading fails with errors like `No module named 'numpy._core'`, `batch_shape unrecognized`, or `layer expected 2 variables, received 0`.

### The Solution — Raw h5py Weight Storage

Weights are saved as raw numpy arrays using `h5py` directly — completely bypassing Keras serialization and version metadata.

**Export script (run locally):**

```python
import tensorflow as tf
from tensorflow import keras
import h5py

old = keras.models.load_model('BPLD_CNN_model.h5', compile=False)

with h5py.File('weights_raw.h5', 'w') as f:
    for i, layer in enumerate(old.layers):
        w = layer.get_weights()
        if len(w) > 0:
            grp = f.create_group(f'layer_{i}')
            for j, arr in enumerate(w):
                grp.create_dataset(f'w{j}', data=arr)

print("Done — upload weights_raw.h5 to Google Drive")
```

**Loading in `main.py`:**

```python
model = keras.Sequential([...])
model(np.zeros((1, 224, 224, 3), dtype=np.float32))  # force build

with h5py.File(MODEL_PATH, 'r') as f:
    for i, layer in enumerate(model.layers):
        key = f'layer_{i}'
        if key in f:
            w = [f[key][k][()] for k in sorted(f[key].keys())]
            layer.set_weights(w)
```

### Swapping the Model

1. Run the export script above with your new trained model
2. Upload `weights_raw.h5` to Google Drive
3. Update `MODEL_DRIVE_ID` in Render environment variables
4. Redeploy — weights download automatically

If your new model has a different architecture, also update the `keras.Sequential([...])` block in `main.py` and `CLASS_NAMES` if classes changed.

---

## 📊 Results & Performance

### CNN Model Performance (Clean Data)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anthracnose | 0.94 | 0.93 | 0.93 |
| Healthy | 0.98 | 0.97 | 0.97 |
| Leaf Crinkle | 0.87 | 0.88 | 0.87 |
| Powdery Mildew | 0.91 | 0.90 | 0.90 |
| Yellow Mosaic | 0.90 | 0.89 | 0.89 |
| **Overall** | | | **91.4%** |

### Defense Effectiveness

```
Clean accuracy:          91.4%
Under FGSM (defended):   79.2%  (+51% vs undefended 28%)
Under PGD  (defended):   71.0%  (+60% vs undefended 11%)
Under C&W  (defended):   68.3%  (+49% vs undefended 19%)
```

---

## 🌐 AgriDoc — Multilingual AI Chat

Supported languages: 🇬🇧 English | 🇮🇳 हिंदी | मराठी | తెలుగు | தமிழ் | ಕನ್ನಡ | বাংলা | ગુજરાતી | ਪੰਜਾਬੀ

Features: voice input, text-to-speech replies, photo upload in chat, save to Farm Diary.

---

## 📚 References

1. Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples* (FGSM)
2. Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD)
3. Carlini & Wagner (2017) — *Evaluating the Robustness of Neural Networks* (C&W)
4. Mohanty et al. (2016) — *Using Deep Learning for Image-Based Plant Disease Detection*
5. Ferentinos (2018) — *Deep Learning Models for Plant Disease Detection and Diagnosis*
6. Brahimi et al. (2020) — *Adversarial Attacks on Plant Disease Classifiers*

---

## 👤 Done By

**Abdul Samad Shaikh**
Bachelor of Engineering — Information Technology
2024–25

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with 🌿 for Indian farmers

⭐ Star this repo if you found it useful!

</div>
