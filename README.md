# DriverGuard — Driver Behavior Classification System

An AI-powered system that classifies driver behavior from images using deep learning. Built with PyTorch, FastAPI, and React.

---

## Overview

DriverGuard detects distracted driving behaviors from images and assigns a risk level to each detection. It supports two production-ready models trained on the State Farm Distracted Driver dataset.

### Detection Classes

| ID | Behavior | Risk |
|----|----------|------|
| 0 | Safe Driving | Safe |
| 1 | Texting (Right Hand) | High |
| 2 | Talking on Phone (Right) | High |
| 3 | Texting (Left Hand) | High |
| 4 | Talking on Phone (Left) | High |
| 5 | Operating Radio | Medium |
| 6 | Drinking | Medium |
| 7 | Reaching Behind | Medium |
| 8 | Hair & Makeup | Low |
| 9 | Talking to Passenger | Low |

---

## Models

| Model | Val Accuracy | Test Accuracy |
|-------|-------------|---------------|
| EfficientNet-B3 | 98.88% | **99.15%** |
| ResNet50 | 99.15% | 99.02% |

Both models were trained for 30 epochs with data augmentation, batch normalization, label smoothing, and learning rate scheduling.

---

## Project Structure

```
Driver_Gesture_Detection_System/
├── config/
│   ├── training.yaml        # Training hyperparameters
│   ├── model.yaml           # Model architecture config
│   └── class_mapping.yaml   # Class index → name mapping
├── checkpoints/             # Model weights (not tracked in git)
│   ├── efficientnet_b3_best.pth
│   └── resnet50_best.pth
├── src/
│   ├── main.py              # FastAPI application
│   ├── prediction.py        # Inference pipeline
│   ├── preprocess.py        # Image preprocessing
│   ├── train_classifier.py  # Model training logic
│   └── evaluate.py          # Evaluation utilities
├── frontend/
│   ├── src/app/App.tsx      # React dashboard
│   ├── Dockerfile
│   └── nginx.conf
├── notebooks/
│   └── pipeline.ipynb       # Google Colab training notebook
├── docs/screenshots/        # App screenshots
├── results/                 # Training metrics and visualizations
├── Dockerfile               # Backend image
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (Docker)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

### Run

```bash
git clone https://github.com/Hassan-essoufi/Driver-Gesture-Classification-System
cd Driver_Gesture_Detection_System
```

> **Note:** Model checkpoints are not included in the repo due to file size. Download them and place them in `checkpoints/`:
> - `checkpoints/resnet50_best.pth`
> - `checkpoints/efficientnet_b3_best.pth`

```bash
docker compose up --build
```

- Frontend → `http://localhost`
- Backend API → `http://localhost:8000`
- API Docs → `http://localhost:8000/docs`

### Stop

```bash
docker compose down
```

---

## Local Development (without Docker)

### Backend

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run
cd src
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## API

### `POST /predict`

Classify a driver image.

**Query params:**
- `model_name` — `resnet50` or `efficientnet_b3`

**Body:** `multipart/form-data` with `file` field (JPG, PNG, WebP — max 5MB)

**Response:**
```json
{
  "class": "Safe Driving",
  "label_id": 0,
  "confidence": 0.9921
}
```

### `GET /`

Health check — returns `{"status": "ok"}`.

---

## Training Techniques

### Optimizer
- **AdamW** — Adam with decoupled weight decay
  - Learning rate: `0.0003`
  - Weight decay: `0.0005`

### Learning Rate Scheduling
- **ReduceLROnPlateau** — reduces LR when validation loss stops improving
  - Factor: `0.2` (multiply LR by 0.2 on plateau)
  - Patience: `3` epochs before reducing
  - Minimum LR: `1e-7`

### Regularization

| Technique | Details |
|-----------|---------|
| **Dropout** | 0.4 in the classifier head (both models) |
| **Batch Normalization** | Applied after the hidden layer in the classifier |
| **Weight Decay** | L2 regularization via AdamW (`0.0005`) |
| **Label Smoothing** | `0.1` — prevents overconfident predictions |
| **Early Stopping** | Stops training if val accuracy doesn't improve for `7` epochs |
| **Frozen Backbone** | Pretrained backbone weights frozen — only classifier head trained |

### Data Augmentation (Training only)

| Augmentation | Parameters |
|-------------|------------|
| Random Horizontal Flip | p = 0.5 |
| Random Rotation | ±10° |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 |
| ImageNet Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### Transfer Learning
Both models are initialized with **ImageNet pretrained weights**. The backbone is frozen and only the custom classifier head is trained:
```
Pretrained Backbone → Flatten → Linear(512) → BatchNorm → ReLU → Dropout(0.4) → Linear(10)
```

---

## Results

### Metrics Comparison

| Metric | ResNet50 | EfficientNet-B3 | Best |
|--------|----------|-----------------|------|
| Accuracy | 99.02% | **99.15%** | EfficientNet-B3 |
| Precision | 99.05% | **99.13%** | EfficientNet-B3 |
| Recall | 98.96% | **99.13%** | EfficientNet-B3 |
| F1-Score | 99.00% | **99.13%** | EfficientNet-B3 |

**Best model: EfficientNet-B3** (average score: 99.14%)

### Visualizations

**Confusion Matrices**
![Confusion Matrices](results/visualizations/20260415_085013_confusion_matrices.png)

**Global Metrics Comparison**
![Global Metrics](results/visualizations/20260415_085013_global_metrics_comparison.png)

**Accuracy per Class**
![Accuracy per Class](results/visualizations/20260415_085013_accuracy_per_class.png)

**Train / Val / Test Phase Comparison**
![Phase Comparison](results/visualizations/20260415_085013_phase_comparison.png)

---

## Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Dashboard</b></td>
      <td align="center"><b>Analyze — Drinking · Medium Risk</b></td>
      <td align="center"><b>Analyze — Safe Driving</b></td>
      <td align="center"><b>History</b></td>
    </tr>
    <tr>
      <td><img src="docs/screenshots/Screenshot 2026-04-16 065223.png" width="380"/></td>
      <td><img src="docs/screenshots/Screenshot 2026-04-16 065028.png" width="380"/></td>
      <td><img src="docs/screenshots/Screenshot 2026-04-16 065149.png" width="380"/></td>
      <td><img src="docs/screenshots/Screenshot 2026-04-16 065254.png" width="380"/></td>
    </tr>
  </table>
</div>

---

## Training (Google Colab)

Open `notebooks/pipeline.ipynb` in Google Colab with GPU runtime.

The notebook:
1. Mounts Google Drive
2. Copies data to local SSD for fast I/O
3. Trains both models with early stopping
4. Evaluates on test set and saves checkpoints to Drive

After training, download the `.pth` files and place them in `checkpoints/`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `RELOAD` | `False` | Auto-reload (dev only) |
| `THRESHOLD` | `0.7` | Minimum confidence threshold |
| `MAX_FILE_SIZE_MB` | `5` | Max upload size in MB |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

---

## Tech Stack

- **Backend** — Python, FastAPI, PyTorch, Uvicorn
- **Frontend** — React, TypeScript, Vite, Tailwind CSS, Recharts
- **Models** — EfficientNet-B3, ResNet50 (torchvision)
- **Deployment** — Docker, Docker Compose, nginx

---

## Author

**Hassan Essoufi**
 Data & AI engineer

---

## ⭐ Note

If you find this project useful, consider giving it a ⭐ — it helps a lot!
