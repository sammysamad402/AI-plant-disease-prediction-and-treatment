# 🌿 PlantDoc AI — Robust Defense System Against Adversarial Attacks

<div align="center">

![PlantDoc AI Banner](https://img.shields.io/badge/PlantDoc-AI%20Agriculture%20Platform-1B6B3A?style=for-the-badge&logo=leaf&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)

**AI-powered plant disease detection with adversarial attack defense, multilingual expert chat, and full farm management tools.**

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results--performance) • [Team](#-team)

</div>

---

## 📌 About The Project

PlantDoc AI is a full-stack agricultural intelligence platform that detects plant diseases from leaf images using a custom-trained CNN, while actively defending against adversarial attacks — invisible pixel manipulations that fool standard AI models.

> **Problem:** Existing plant disease AI systems achieve high accuracy in lab settings but are completely undefended against adversarial noise. A perturbation invisible to the human eye can drop model accuracy from 91% to 11%.

> **Our Solution:** A 5-layer defense pipeline implementing FGSM, PGD, and C&W attack simulation with Gaussian blur and median filter defenses — recovering up to 79% accuracy under attack.

Built as a **Major Project** for Computer Engineering, guided by **Prof. Farzana Khan**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **Disease Detection** | CNN model with 91.4% accuracy across 5 disease classes |
| 🛡️ **Adversarial Defense** | FGSM, PGD & C&W attack simulation + 5-layer defense pipeline |
| 🌦️ **Weather Enhancement** | Auto-corrects foggy, dark, and low-contrast field images |
| 📹 **Live Webcam** | Real-time detection every 5 seconds with auto-defense |
| 💬 **AgriDoc AI Chat** | Multilingual expert AI in 9 Indian languages with voice I/O |
| 📔 **Farm Diary** | Complete detection & consultation history with tags and filters |
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
│  Layer 1 → Plant Verification (HSV Analysis)    │
│  Layer 2 → Weather Enhancement (PIL Filters)    │
│  Layer 3 → Adversarial Detection (Sobel Grad.)  │
│  Layer 4 → Signal Smoothing (Gaussian + Median) │
│  Layer 5 → Confidence Gating (≥60% threshold)   │
└─────────────────────────────────────────────────┘
    │
    ▼
BPLD CNN Model (224×224 input → 5-class Softmax)
    │
    ▼
Disease Name + Confidence + Severity + Treatment
```

### Adversarial Attacks Tested

| Attack | Method | Accuracy Without Defense | Accuracy With Defense |
|---|---|---|---|
| FGSM | Fast Gradient Sign (ε=0.03) | 28% | **79.2%** |
| PGD | Projected Gradient Descent (5 iters) | 11% | **71.0%** |
| C&W | Carlini & Wagner (L2 optimize) | 19% | **68.3%** |

---

## 🛠️ Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework
- [TensorFlow / Keras](https://tensorflow.org) — CNN model training & inference
- [OpenCV](https://opencv.org) — Image processing & adversarial detection
- [Pillow (PIL)](https://pillow.readthedocs.io) — Weather enhancement filters
- [MongoDB](https://mongodb.com) — Database for detections & diary
- [NumPy](https://numpy.org) — Numerical operations

**Frontend**
- HTML5, CSS3, JavaScript — Unified web UI
- Inter font — consistent typography across all pages

**AI Services**
- [OpenAI GPT-4o-mini](https://openai.com) — AgriDoc multilingual expert chat
- Web Speech API — voice input and text-to-speech output

**DevOps**
- [Uvicorn](https://uvicorn.org) — ASGI production server
- python-dotenv — environment variable management

---

## 📁 Project Structure

```
plantdoc-ai/
│
├── main.py              # FastAPI app — detection, records, home/predict/webcam pages
├── consultation.py      # AgriDoc AI chat — multilingual expert consultation
├── diary.py             # Farm diary + agri shops page
├── crop_calendar.py     # Month-by-month crop farming guide
├── visualization.py     # Data visualization & analytics
├── shared_ui.py         # Unified design system — CSS, navbar, shared components
│
├── BPLD_CNN_model.h5    # Trained CNN model (not included — see below)
├── uploads/             # Uploaded & processed images (auto-created)
├── static/              # Static assets
│
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- MongoDB (local or [Atlas](https://mongodb.com/atlas))
- OpenAI API key (for AgriDoc chat)

### 1. Clone the repository

```bash
git clone https://github.com/AbdulSamad750/AI-plant-disease-prediction-and-treatment.git
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
MODEL_PATH=BPLD_CNN_model.h5
UPLOAD_DIR=uploads
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Add the trained model

Place your `BPLD_CNN_model.h5` file in the root directory.  
> The model file is not included in this repo due to size. Contact the team or train your own using the BPLD dataset.

### 6. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000**

---

## 🚀 Usage

### Web Application

| Page | URL | Description |
|---|---|---|
| Home | `/` | Project overview and navigation |
| Detect | `/predict` | Upload leaf photo for disease detection |
| Live Cam | `/webcam` | Real-time webcam detection |
| Records | `/records-page` | View all past detections |
| Consult | `/consultation` | Chat with AgriDoc AI |
| Diary | `/diary` | Farm diary and history |
| Calendar | `/crop-calendar` | Crop seasonal guide |
| Agri Shops | `/agri-shops` | Find nearby stores |

### API Endpoints

```http
POST   /detect              # Upload image for disease detection
GET    /records             # Fetch detection history
DELETE /records/{id}        # Delete a record
GET    /image/{filename}    # Serve uploaded image
POST   /ask-expert          # Send message to AgriDoc AI
POST   /diary/save          # Save diary entry
GET    /diary/entries       # Get all diary entries
GET    /crop-calendar       # Crop calendar page
```

### Raspberry Pi / Field Camera

```bash
python raspiclient.py --server http://your-server:8000 --device field-cam-1 --interval 10
```

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

Supported languages for expert consultation:

🇬🇧 English &nbsp;|&nbsp; 🇮🇳 हिंदी &nbsp;|&nbsp; मराठी &nbsp;|&nbsp; తెలుగు &nbsp;|&nbsp; தமிழ் &nbsp;|&nbsp; ಕನ್ನಡ &nbsp;|&nbsp; বাংলা &nbsp;|&nbsp; ગુજરાતી &nbsp;|&nbsp; ਪੰਜਾਬੀ

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

## 👥 Team

| Name | Roll No |
|---|---|
| Abdul Samad Shaikh | 221435 |
| Shubham Gupta | 221414 |
| Shaikh Anas | 221437 |
| Shaikh Sana | 221451 |

**Guided by:** Prof. Farzana Khan  
**Department:** Computer Engineering  
**Year:** 2024–25

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with 🌿 for Indian farmers

⭐ Star this repo if you found it useful!

</div>
