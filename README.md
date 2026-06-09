<div align="center">

# 🌿 PlantDoc AI

### AI-Powered Plant Disease Detection with Adversarial Defense

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

<br/>

[![Published in IJARSCT](https://img.shields.io/badge/Published-IJARSCT%20%7C%20Vol.6%20Issue%2019%20%7C%20Apr%202026-1a73e8?style=for-the-badge&logo=googlescholar&logoColor=white)](https://ijarsct.co.in/Paper34436.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.48175%2FIJARSCT--34436-orange?style=for-the-badge)](https://doi.org/10.48175/IJARSCT-34436)
[![Impact Factor](https://img.shields.io/badge/Impact%20Factor-8.2-brightgreen?style=for-the-badge)](https://ijarsct.co.in/Apr6i19.html)
[![Analytics Dashboard](https://img.shields.io/badge/Analytics%20Dashboard-Live%20Deployment-7B61FF?style=for-the-badge\&logo=plotly\&logoColor=white)](https://sammysamad402-plantdoc-ai-analytics.hf.space)
[![Behance Showcase](https://img.shields.io/badge/Behance-Featured%20Case%20Study-1769FF?style=for-the-badge\&logo=behance\&logoColor=white)](https://www.behance.net/gallery/250834343/AI-Powered-Plant-Disease-Detection-Analytics-Platform)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-PlantDoc%20AI-00C853?style=for-the-badge\&logo=huggingface\&logoColor=white)](https://sammysamad402-plantdoc-ai.hf.space)

<br/>

A full-stack agricultural intelligence platform that detects plant diseases from leaf images using a custom-trained CNN, with a 5-layer defense pipeline against adversarial attacks, JWT-based authentication, and a multilingual expert consultation system — **peer-reviewed and published in an international journal.**

<br/>

**[🚀 Live Demo](https://sammysamad402-plantdoc-ai.hf.space)** &nbsp;·&nbsp; **[📄 Read the Paper](https://ijarsct.co.in/Paper34436.pdf)** &nbsp;·&nbsp; **[🌐 Journal Issue](https://ijarsct.co.in/Apr6i19.html)**

</div>

---

## 📰 Research Publication

> This project was formally accepted, peer-reviewed, and published in an international research journal.

<div align="center">

| | |
|:---|:---|
| 📌 **Title** | WeatherLeaf: Multi-Modal Plant Disease Risk Prediction by Fusing Leaf Images with Real-Time Meteorological Data for Indian Farms |
| 📖 **Journal** | International Journal of Advanced Research in Science, Communication and Technology (IJARSCT) |
| 🔖 **ISSN** | 2581-9429 &nbsp;·&nbsp; Open-Access, Double-Blind, Peer-Reviewed |
| 📅 **Published** | Volume 6, Issue 19 — April 2026 |
| 🔗 **DOI** | [10.48175/IJARSCT-34436](https://doi.org/10.48175/IJARSCT-34436) |
| ⭐ **Impact Factor** | 8.2 (RPRI Indexed) &nbsp;·&nbsp; Crossref Registered |
| 🏫 **Institution** | M. H. Saboo Siddik College of Engineering, Mumbai, Maharashtra, India |

</div>

<br/>

📄 &nbsp;[**Read Full Paper (PDF)**](https://ijarsct.co.in/Paper34436.pdf) &nbsp;&nbsp;|&nbsp;&nbsp; 🌐 &nbsp;[**View Journal Issue**](https://ijarsct.co.in/Apr6i19.html) &nbsp;&nbsp;|&nbsp;&nbsp; 🔗 &nbsp;[**DOI Link**](https://doi.org/10.48175/IJARSCT-34436)

---

## Overview

PlantDoc AI addresses a critical gap in agricultural AI: most plant disease detection models perform well in controlled lab settings but are completely vulnerable to adversarial perturbations — pixel-level noise invisible to the human eye that can collapse model accuracy from 91% to as low as 11%.

This project implements a complete defense pipeline alongside a production-grade web application with user authentication, farm management tools, and multilingual expert AI chat — designed for real-world use by Indian farmers.

**Key outcomes:**
- 91.4% baseline accuracy on a 5-class plant disease dataset
- Adversarial accuracy recovered to 79.2% under FGSM attack (vs. 28% undefended)
- Secure multi-user platform with private per-user data isolation
- Multilingual AI consultation in 9 Indian languages

---

## Features

| Feature | Description |
|---|---|
| 🔬 **Disease Detection** | CNN model classifying 5 plant disease categories with 91.4% accuracy |
| 🛡️ **Adversarial Defense** | 5-layer pipeline defending against FGSM, PGD, and C&W attacks |
| 🔐 **JWT Authentication** | Secure register/login system; every user's data is fully private |
| 📹 **Live Webcam Detection** | Real-time inference every 5 seconds with auto-defense enabled |
| 💬 **AgriDoc AI Chat** | GPT-4o-mini powered expert in 9 Indian languages with voice I/O |
| 🌦️ **Weather Enhancement** | Auto-correction for foggy, dark, and low-contrast field images |
| 📔 **Farm Diary** | Personal detection history and consultation log with tags and filters |
| 🗓️ **Crop Calendar** | Month-by-month farming guide for 5 major Indian crops |
| 🏪 **Agri Shops Map** | GPS-based locator for nearby agricultural stores and KVK centers |

---

## Detectable Diseases

| Disease | Severity | Description |
|---|---|---|
| Anthracnose | 🔴 High | Dark fungal lesions on leaves and stems |
| Healthy | ✅ None | Vibrant green foliage, no disease signs |
| Leaf Crinkle | 🟡 Medium | Viral leaf distortion and puckering |
| Powdery Mildew | 🟡 Medium | White powder coating on leaf surfaces |
| Yellow Mosaic | 🔴 High | Viral yellow mottling patterns |

---

## System Architecture

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
BPLD CNN Model (224×224 → 5-class Softmax)
    │
    ▼
Disease Name + Confidence + Severity + Treatment
```

### CNN Architecture

```
Input (224×224×3)
→ Conv2D(32, 3×3, ReLU)  → MaxPool
→ Conv2D(64, 3×3, ReLU)  → MaxPool
→ Conv2D(128, 3×3, ReLU) → MaxPool
→ Flatten
→ Dense(128, ReLU) → Dropout(0.5)
→ Dense(5, Softmax)
```

### Adversarial Robustness

| Attack | Method | Undefended | Defended | Recovery |
|---|---|---|---|---|
| FGSM | Fast Gradient Sign (ε=0.03) | 28% | **79.2%** | +51% |
| PGD | Projected Gradient Descent (5 iters) | 11% | **71.0%** | +60% |
| C&W | Carlini & Wagner (L2 optimize) | 19% | **68.3%** | +49% |

---

## Authentication & Security

PlantDoc AI uses JWT (JSON Web Token) based authentication with the following security practices:

- Passwords are hashed using **bcrypt** before storage — plaintext passwords are never stored
- Login returns a signed JWT token with a 7-day expiry, stored client-side
- Every API request authenticates via `Authorization: Bearer <token>` header
- All detection records, diary entries, and consultations are scoped to the authenticated `user_id`
- Secrets (JWT key, database URI, API keys) are managed exclusively through environment variables — never hardcoded

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework with async support
- [TensorFlow / Keras 2.13](https://tensorflow.org) — CNN model inference
- [OpenCV](https://opencv.org) — Image processing and adversarial detection
- [Pillow (PIL)](https://pillow.readthedocs.io) — Weather enhancement filters
- [MongoDB Atlas](https://mongodb.com) — Cloud database for users, detections, and diary
- [python-jose](https://github.com/mpdavis/python-jose) — JWT token creation and verification
- [passlib + bcrypt](https://passlib.readthedocs.io) — Secure password hashing
- [h5py](https://www.h5py.org) — Cross-version-safe model weight storage
- [gdown](https://github.com/wkentaro/gdown) — Automatic model download from Google Drive

**Frontend**
- HTML5, CSS3, vanilla JavaScript — served directly from FastAPI static routes

**AI Services**
- [OpenAI GPT-4o-mini](https://openai.com) — AgriDoc multilingual expert chat
- Web Speech API — voice input and text-to-speech output

**Infrastructure**
- [Hugging Face Spaces](https://huggingface.co/spaces) — Cloud deployment via Docker
- [Docker](https://docker.com) — Containerized runtime environment
- [Google Drive + gdown](https://github.com/wkentaro/gdown) — Model weight hosting and auto-download at startup
- [Uvicorn](https://uvicorn.org) — ASGI production server

---

## Project Structure

```
AI-plant-disease-prediction-and-treatment/
│
├── main.py              # FastAPI app — detection, records, all page routes
├── auth.py              # JWT authentication — register, login, token verification
├── consultation.py      # AgriDoc AI chat — multilingual expert consultation
├── diary.py             # Farm diary and agri shops page
├── crop_calendar.py     # Month-by-month crop farming guide
├── visualization.py     # Data visualization and analytics
├── shared_ui.py         # Unified design system — CSS, navbar, auth helpers
├── Dockerfile           # Container definition for Hugging Face Spaces
│
├── requirements.txt     # Python dependencies
├── .python-version      # Python version pin (3.11)
├── .gitignore
└── README.md
```

> **Note:** The trained model file (`weights_raw.h5`) is intentionally excluded from this repository due to its size. It is automatically downloaded from Google Drive at container startup using `gdown`. See [Model Weights](#model-weights) below.

---

## Installation

### Prerequisites

- Python 3.11
- MongoDB (local instance or [MongoDB Atlas](https://mongodb.com/atlas))
- OpenAI API key (for AgriDoc consultation)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/sammysamad402/AI-plant-disease-prediction-and-treatment.git
cd AI-plant-disease-prediction-and-treatment

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (see section below)
cp .env.example .env
# Edit .env with your values

# 5. Run the development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000** — you will be redirected to register or log in.

---

## Environment Variables

Create a `.env` file in the project root with the following values:

```env
# Database
MONGO_URI=mongodb://localhost:27017/

# Model
MODEL_PATH=weights_raw.h5
MODEL_DRIVE_ID=your_google_drive_file_id

# File storage
UPLOAD_DIR=uploads

# AI services
OPENAI_API_KEY=your_openai_api_key

# Authentication
JWT_SECRET_KEY=your-long-random-secret-key
JWT_EXPIRE_MINUTES=10080
```

| Variable | Description |
|---|---|
| `MONGO_URI` | MongoDB connection string (local or Atlas) |
| `MODEL_DRIVE_ID` | Google Drive file ID of the model weights |
| `OPENAI_API_KEY` | OpenAI API key for AgriDoc chat |
| `JWT_SECRET_KEY` | A long, random secret string for signing tokens |
| `JWT_EXPIRE_MINUTES` | Token expiry in minutes — default is `10080` (7 days) |

---

## Deployment on Hugging Face Spaces

PlantDoc AI is deployed on [Hugging Face Spaces](https://huggingface.co/spaces) using a Docker runtime.

### Why Hugging Face Spaces?

The application was initially prototyped on **Render's free tier**. During load testing, TensorFlow inference workloads consistently exceeded Render's free-tier memory limits (~512 MB), causing the container to restart mid-request. Hugging Face Spaces provides a Docker-based environment with sufficient memory headroom for TensorFlow model inference, making it the appropriate host for this application.

### How the Docker deployment works

On container startup, `main.py` checks for the model weights file. If it is not present (as is the case on a fresh container), it downloads automatically from Google Drive:

```python
if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )
```

This keeps large binary files out of the repository while ensuring the deployed container is always self-sufficient.

### Deploying to Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces), selecting **Docker** as the SDK
2. Push your repository to the Space (or link your GitHub repo)
3. Set the following environment variables in the Space **Settings → Variables and Secrets**:

| Variable | Value |
|---|---|
| `MONGO_URI` | Your MongoDB Atlas connection string |
| `MODEL_DRIVE_ID` | Google Drive file ID of `weights_raw.h5` |
| `MODEL_PATH` | `weights_raw.h5` |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `JWT_SECRET_KEY` | A long random secret string |
| `JWT_EXPIRE_MINUTES` | `10080` |

4. The Space will build the Docker image and deploy automatically.

---

## Model Weights

The model was trained locally and exported as raw numpy arrays using `h5py` — bypassing Keras serialization entirely. This solves cross-version compatibility issues where models trained on newer Keras versions fail to load on the deployment environment's TensorFlow 2.13.

**To update the model:**

1. Train your new model and export weights:

```python
import h5py
from tensorflow import keras

model = keras.models.load_model('your_new_model.h5', compile=False)

with h5py.File('weights_raw.h5', 'w') as f:
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            grp = f.create_group(f'layer_{i}')
            for j, arr in enumerate(weights):
                grp.create_dataset(f'w{j}', data=arr)
```

2. Upload `weights_raw.h5` to Google Drive (replace the existing file or create a new one)
3. Update `MODEL_DRIVE_ID` in your Space secrets
4. Restart the Space — weights will re-download automatically

---

## API Endpoints

### Authentication

```
POST  /auth/register   # Create new account
POST  /auth/login      # Login — returns signed JWT token
```

### Detection

```
POST   /detect              # Upload image for disease detection
GET    /records             # Fetch current user's detection history
DELETE /records/{id}        # Delete a detection record (own records only)
GET    /image/{filename}    # Serve uploaded image file
```

### Consultation & Diary

```
POST  /ask-expert       # Send message to AgriDoc AI
POST  /diary/save       # Save a diary entry
GET   /diary/entries    # Fetch current user's diary entries
```

### Pages (Auth required except login/register)

```
GET  /              → Home
GET  /predict       → Upload & detect
GET  /webcam        → Live webcam detection
GET  /records-page  → Detection history
GET  /consultation  → AgriDoc AI chat
GET  /diary         → Farm diary
GET  /crop-calendar → Crop calendar
GET  /agri-shops    → Agricultural store locator
GET  /login         → Login page (public)
GET  /register      → Register page (public)
```

---

## Results & Performance

### Classification Report (Clean Data)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anthracnose | 0.94 | 0.93 | 0.93 |
| Healthy | 0.98 | 0.97 | 0.97 |
| Leaf Crinkle | 0.87 | 0.88 | 0.87 |
| Powdery Mildew | 0.91 | 0.90 | 0.90 |
| Yellow Mosaic | 0.90 | 0.89 | 0.89 |
| **Overall** | | | **91.4%** |

### Defense Effectiveness Summary

```
Clean accuracy:                91.4%
Under FGSM attack (defended):  79.2%   (vs. 28% undefended)
Under PGD  attack (defended):  71.0%   (vs. 11% undefended)
Under C&W  attack (defended):  68.3%   (vs. 19% undefended)
```

---

## AgriDoc — Multilingual AI Consultation

AgriDoc is an in-app agricultural expert powered by GPT-4o-mini. It supports voice input, text-to-speech responses, image upload within the chat, and direct saving of consultations to the Farm Diary.

**Supported languages:** English · हिंदी · मराठी · తెలుగు · தமிழ் · ಕನ್ನಡ · বাংলা · ગુજરાતી · ਪੰਜਾਬੀ

---

## Future Improvements

- Expand disease coverage beyond 5 classes using a larger annotated dataset
- Add offline model inference via TensorFlow Lite for low-connectivity environments
- Integrate government scheme and subsidy data into the consultation system
- Push notification alerts for seasonal disease outbreaks based on GPS location
- Admin dashboard for aggregated farm health analytics across users

---

## References

1. Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples* (FGSM)
2. Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD)
3. Carlini & Wagner (2017) — *Evaluating the Robustness of Neural Networks* (C&W)
4. Mohanty et al. (2016) — *Using Deep Learning for Image-Based Plant Disease Detection*
5. Ferentinos (2018) — *Deep Learning Models for Plant Disease Detection and Diagnosis*
6. Brahimi et al. (2020) — *Adversarial Attacks on Plant Disease Classifiers*

---

## Author

<div align="center">

**Abdul Samad Shaikh**

Bachelor of Engineering — Information Technology
M. H. Saboo Siddik College of Engineering, Mumbai · 2024–25

<br/>

📰 &nbsp;**Published Researcher** — IJARSCT Vol. 6, Issue 19, April 2026

*WeatherLeaf: Multi-Modal Plant Disease Risk Prediction by Fusing Leaf Images with Real-Time Meteorological Data for Indian Farms*

[![DOI](https://img.shields.io/badge/DOI-10.48175%2FIJARSCT--34436-orange?style=flat-square)](https://doi.org/10.48175/IJARSCT-34436)
[![Read Paper](https://img.shields.io/badge/Read%20Paper-PDF-red?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://ijarsct.co.in/Paper34436.pdf)

</div>

---
## 🌐 Live Deployments

### 🌿 PlantDoc AI
https://sammysamad402-plantdoc-ai.hf.space

### 📊 PlantDoc Analytics Dashboard
https://sammysamad402-plantdoc-ai-analytics.hf.space

---

### 🎨 Project Showcase:
https://www.behance.net/gallery/250834343/AI-Powered-Plant-Disease-Detection-Analytics-Platform

---

## License

---

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with 🌿 for Indian farmers

⭐ &nbsp;If you found this useful, consider starring the repo

</div>
