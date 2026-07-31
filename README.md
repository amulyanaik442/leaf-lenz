# Leaf Lenz

Leaf Lenz is an AI-driven plant disease detection web application built with Django and ONNX. It allows users to upload images or take live photos of plant leaves to instantly identify diseases, receive a confidence score, and get actionable treatment plans.

## Features

- **Multi-Model Architecture:** Uses a general 90-class model plus dedicated crop-specific models for Wheat (11 diseases) and Maize (4 diseases).
- **Crop Selection:** Users can select their crop (Wheat, Maize, or Auto-detect) for targeted, accurate predictions.
- **Smart Routing:** When set to Auto, routes images to the most appropriate model based on confidence thresholds.
- **Test-Time Augmentation (TTA):** 20-view TTA for general model, 5-crop TTA for wheat/maize — boosts real-world accuracy.
- **Lightning Fast Inference:** Optimized ONNX models running on CPU for instant prediction without GPU.
- **Dynamic UI Engine:** Real-time visual feedback with Red (Disease), Green (Healthy), and Grey (Invalid Image) alert banners.
- **Encyclopedia & Analytics:** Built-in encyclopedia of diseases and a dashboard to track scan statistics.
- **Live Camera Support:** Built-in support for capturing leaf images directly from your device's camera.

## Models

| Model | Classes | Architecture | Accuracy | File |
|-------|---------|-------------|----------|------|
| General | 90 crop diseases | MobileNetV2 | ~82.4% | `detector/ml_assets/model.onnx` |
| Wheat | 11 diseases | MobileNetV2 | 81.3% | `detector/ml_assets/wheat/wheat_model.onnx` |
| Maize | 4 diseases | MobileNetV2 | 92.8% | `detector/ml_assets/maize/maize_model.onnx` |

### Wheat Disease Classes (11)
Blast, Common Root Rot, Fusarium Head Blight, Healthy, Leaf Spot, Mildew, Pest, Rust, Septoria, Smut, Stem fly

> **Note:** Original 15 classes were consolidated via visual similarity analysis:
> - Black Rust + Brown Rust + Yellow Rust → **Rust**
> - Leaf Blight + Tan spot → **Leaf Spot**
> - Aphid + Mite → **Pest**

### Maize Disease Classes (4)
Blight, Common Rust, Gray Leaf Spot, Healthy

### General Model Classes (90)
Apple, Cassava, Cherry, Citrus, Coffee, Corn, Cotton, Grape, Pepper, Potato, Rice, Tomato, and more — covering 90 crop-disease combinations.

### Routing Logic
1. If user selects **Wheat** → wheat model directly
2. If user selects **Maize** → maize model directly
3. If **Auto-detect** → wheat first (confidence >= 10%), then maize (>= 50%), then general fallback

## Project Structure

- `detector/`: Main Django app containing views, encyclopedia logic, and ONNX inference.
- `detector/ml_assets/`: Deployed ONNX models and class name files.
  - `model.onnx` - General 90-class model
  - `wheat/wheat_model.onnx` - Wheat-specific model (11 classes)
  - `maize/maize_model.onnx` - Maize-specific model (4 classes)
- `detector/predictors/`: Crop-specific prediction modules.
  - `wheat_predictor.py` - Wheat ONNX inference (5-crop TTA)
  - `maize_predictor.py` - Maize ONNX inference (5-crop TTA, temperature scaling)
- `detector/inference.py` - Smart routing and crop-aware prediction
- `detector/disease_info.py` - Shared disease data loader
- `leaf_lenz_project/`: Core Django project configuration.
- `ml_model/`: Scripts for training, evaluation, and dataset management.

## Prerequisites

- Python 3.8+
- Git

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/amulyanaik442/leaf-lenz.git
cd leaf-lenz
```

### 2. Set up a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

```bash
cp .env.example .env
```

Set the `SECRET_KEY` and other variables in `.env`.

### 5. Database Setup

```bash
python manage.py migrate
```

### 6. Run the Application

```bash
python manage.py runserver
```

The application will be accessible at `http://127.0.0.1:8000/`.

## Training Crop-Specific Models

Each crop model is trained independently without modifying the original project.

### Wheat Model

```bash
python ml_model/download_wheat_dataset.py    # Download from Kaggle
python ml_model/rebalance_wheat.py           # Balance dataset
python ml_model/train_wheat.py               # Train MobileNetV2
python ml_model/evaluate_wheat.py            # Evaluate on test set
python ml_model/export_wheat_onnx.py         # Export to ONNX
```

### Maize Model

```bash
python ml_model/download_maize_dataset.py    # Download from Kaggle
python ml_model/prepare_maize_dataset.py     # Prepare dataset
python ml_model/balance_maize_dataset.py     # Balance dataset
python ml_model/train_maize.py               # Train MobileNetV2
python ml_model/evaluate_maize.py            # Evaluate on test set
python ml_model/export_maize_onnx.py         # Export to ONNX
```

## Adding New Crops

To add a new crop-specific model, follow the same pattern:
1. Download dataset (Kaggle or custom)
2. Balance via undersampling
3. Train MobileNetV2 (2-phase: head training + fine-tuning)
4. Export to ONNX
5. Create predictor in `detector/predictors/`
6. Add routing logic in `detector/inference.py`
7. Add disease info to `detector/disease_data.json`
