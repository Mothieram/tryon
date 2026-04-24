# 🪡 Virtual Try-On Studio
### AI-Powered Real-Time Shirt Overlay System

> A production-grade virtual dressing room with YOLOv8 pose detection, TPS cloth warping, realistic occlusion, and dynamic lighting — comparable to Myntra, Zara, or Amazon Fashion AR.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [WSL Setup Guide](#wsl-setup-guide)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Adding Custom Shirts](#adding-custom-shirts)
- [GPU Acceleration](#gpu-acceleration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE FLOW                             │
│                                                             │
│  Webcam → Flip → [YOLO Pose Detect] ──→ Keypoints (17pt)  │
│                        ↓                                   │
│             [Human Parsing Engine] ──→ Body Region Masks  │
│                        ↓                                   │
│             [DensePose/Fallback]   ──→ Torso Surface Map  │
│                        ↓                                   │
│  Shirt PNG → [Garment Landmark Detect] → Control Points  │
│                        ↓                                   │
│             [Hybrid TPS Warper]    ──→ Warped Shirt BGRA  │
│                   ↑                                        │
│        Scale + Rotate + TPS + Sleeve Deform               │
│                        ↓                                   │
│             [Occlusion Engine]     ──→ Layer Masks        │
│                        ↓                                   │
│             [Shadow Engine]        ──→ Lighting + Shadow  │
│                        ↓                                   │
│             [Alpha Composite]      ──→ Final Frame        │
└─────────────────────────────────────────────────────────────┘
```

### Engine Modules

| Module | Purpose |
|--------|---------|
| `engine/yolo_pose.py` | YOLOv8 17-keypoint body detection with CUDA + temporal smoothing |
| `engine/densepose_engine.py` | Body surface UV mapping (DensePose or geometric fallback) |
| `engine/parsing_engine.py` | Semantic body part segmentation (SCHP or geometric fallback) |
| `engine/garment_landmarks.py` | Auto-detect shirt control points from PNG assets |
| `engine/hybrid_warper.py` | TPS cloth warping + sleeve deformation + physics lag |
| `engine/occlusion_engine.py` | Arms/head naturally in front of shirt compositing |
| `engine/shadow_engine.py` | Dynamic shadows + lighting color temperature matching |
| `engine/render_pipeline.py` | Master orchestrator with garment catalog and frame caching |
| `ui/app.py` | Premium CustomTkinter desktop GUI |

---

## WSL Setup Guide

### Prerequisites

**Windows 11 / Windows 10 (Build 19041+)** with WSL2 installed.

### Step 1 — Install WSL2 with Ubuntu 22.04

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Restart your PC when prompted.

### Step 2 — Enable WSLg (GUI Support for Camera/Display)

WSLg is built into Windows 11. For Windows 10:

```powershell
# PowerShell (Admin) — update WSL kernel
wsl --update
wsl --shutdown
```

Verify WSLg is working:
```bash
# Inside WSL Ubuntu terminal
echo $DISPLAY
# Should output something like: :0 or :1
```

### Step 3 — Install System Dependencies

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and build tools
sudo apt install -y python3.10 python3.10-venv python3-pip python3.10-dev

# OpenCV dependencies
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# GTK/display for CustomTkinter
sudo apt install -y \
    python3-tk \
    tk-dev \
    tcl-dev \
    libgtk-3-dev \
    libgdk-pixbuf2.0-dev

# Camera support in WSL
sudo apt install -y v4l-utils

# Git, wget
sudo apt install -y git wget curl build-essential
```

### Step 4 — Camera Access in WSL

**Option A: USB Webcam passthrough (recommended)**

1. Install [usbipd-win](https://github.com/dorssel/usbipd-win) on Windows
2. In PowerShell (Admin):
   ```powershell
   winget install --interactive --exact dorssel.usbipd-win
   ```
3. Attach camera to WSL:
   ```powershell
   # List USB devices
   usbipd list
   # Bind your camera (e.g., BUSID 2-5)
   usbipd bind --busid 2-5
   # Attach to WSL
   usbipd attach --wsl --busid 2-5
   ```
4. Verify in WSL:
   ```bash
   ls /dev/video*   # Should show /dev/video0
   v4l2-ctl --list-devices
   ```

**Option B: Use Windows camera app, stream to WSL via VirtualHere or another method**

**Option C: Test with video file instead of webcam**

```python
# In main.py, change camera index to a video file path
cap = cv2.VideoCapture("test_video.mp4")
```

### Step 5 — NVIDIA CUDA in WSL (Optional, for GPU acceleration)

If you have an NVIDIA GPU:

```bash
# Check GPU is visible
nvidia-smi

# If not visible, install CUDA for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2

# Verify CUDA
nvcc --version
```

---

## Installation

### Step 1 — Clone / Copy Project

```bash
# Navigate to your home directory
cd ~

# If using git
git clone <your-repo-url> virtual_tryon
cd virtual_tryon

# OR copy the project folder from Windows
# (WSL can access Windows files at /mnt/c/Users/YourName/...)
cp -r /mnt/c/Users/YourName/Downloads/virtual_tryon ~/virtual_tryon
cd ~/virtual_tryon
```

### Step 2 — Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# Verify Python
python --version  # Should be 3.10.x
pip install --upgrade pip
```

### Step 3 — Install PyTorch (CPU or GPU)

**CPU only (works on any machine):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1 (for NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 4 — Install All Requirements

```bash
pip install -r requirements.txt
```

### Step 5 — Verify Installation

```bash
python -c "
import cv2, numpy, ultralytics, customtkinter, scipy
print('✓ OpenCV:', cv2.__version__)
print('✓ NumPy:', numpy.__version__)
print('✓ Ultralytics:', ultralytics.__version__)
print('✓ scipy:', scipy.__version__)
print('✓ CustomTkinter: OK')
print('All dependencies verified!')
"
```

### Optional: Install DensePose (Advanced, requires CUDA)

```bash
# Install detectron2 for full DensePose support
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install DensePose project
cd /tmp
git clone https://github.com/facebookresearch/detectron2
cd detectron2/projects/DensePose
pip install -e .
```

### Optional: Install ONNX Runtime Parsing Model

```bash
# CPU ONNX
pip install onnxruntime

# GPU ONNX
pip install onnxruntime-gpu

# Download SCHP parsing model (optional, improves accuracy)
# Place in models/ directory
mkdir -p models
# wget <schp-model-url> -O models/schp_atr.onnx
```

---

## Running the Application

### GUI Mode (Full Application)

```bash
source .venv/bin/activate
python main.py
```

On first run, YOLOv8 pose model (~6MB) auto-downloads from Ultralytics CDN.

### Headless Mode (OpenCV window only)

If CustomTkinter fails, the app auto-falls back to OpenCV demo:

```bash
python main.py
```

Controls in headless mode:
- `N` — Next shirt
- `P` — Previous shirt  
- `S` — Take screenshot
- `Q` — Quit

### Using the GUI

1. Launch app → window opens
2. Click **"▶ Start Camera"** in left panel
3. Stand ~1.5–2m from webcam (full torso visible)
4. Shirt overlays automatically
5. Browse shirts in **Wardrobe** panel or use **◀ Prev / Next ▶**
6. Click **📸 Screenshot** to save your try-on

---

## Adding Custom Shirts

### Shirt Asset Requirements

| Property | Value |
|----------|-------|
| Format | PNG with transparency (RGBA) |
| Background | Transparent (alpha = 0) |
| Orientation | Front-facing flat lay |
| Resolution | 400–1000px wide recommended |
| Aspect | Portrait (shirt taller than wide) |

### Placement

```
project/
└── assets/
    └── shirts/
        ├── blue_formal.png
        ├── red_tshirt.png
        ├── striped_shirt.png
        └── your_shirt.png    ← Add here
```

The app auto-detects collar, shoulders, sleeves and hem from your PNG.

### Creating Shirt Assets

**Using GIMP (free):**
1. Open garment image
2. Use "Fuzzy Select" to select white/background
3. Delete background (Edit → Clear)
4. Export as PNG

**Using remove.bg or rembg:**
```bash
pip install rembg
rembg i input_shirt.jpg output_shirt.png
```

---

## GPU Acceleration

### Check GPU Support

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Expected Performance

| Hardware | Expected FPS |
|----------|-------------|
| RTX 3070+ | 45–60+ FPS |
| RTX 2060 / GTX 1080 | 30–45 FPS |
| GTX 1060 / RX 580 | 20–30 FPS |
| Intel i7 CPU only | 12–20 FPS |
| Intel i5 CPU only | 8–15 FPS |

### Switching Models for Speed

Edit `main.py`:

```python
# Fastest (least accurate)
pipeline = RenderPipeline(pose_model="yolov8n-pose.pt")

# Balanced (default)
pipeline = RenderPipeline(pose_model="yolov8s-pose.pt")

# Most accurate (slower)
pipeline = RenderPipeline(pose_model="yolov8m-pose.pt")
```

---

## Performance Tuning

### Reduce CPU load

In `engine/render_pipeline.py`:

```python
# Run parsing every 5 frames instead of 3
self._parse_frame_skip = 5

# Disable TPS warping (use affine only)
self._warper = HybridWarper(tps_smooth=999)  # High smooth = near-affine
```

### Async inference

The pose engine already runs async. For further speedup:

```python
# In YoloPoseEngine, use nano model
pose_model="yolov8n-pose.pt"

# Increase smooth_alpha for faster response (less smoothing)
smooth_alpha=0.6
```

### Frame size

In `ui/app.py`, set webcam resolution:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce from 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce from 720
```

---

## Troubleshooting

### Camera not found
```bash
# Check device
ls /dev/video*
# Grant permissions
sudo chmod 666 /dev/video0
# Test with vlc or ffplay
ffplay /dev/video0
```

### Display not working in WSL
```bash
# Set DISPLAY manually
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1  # Software rendering fallback
python main.py
```

### `customtkinter` not found
```bash
pip install customtkinter
```

### YOLO model download fails
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
mv yolov8n-pose.pt ~/virtual_tryon/
```

### TPS warping crashes (scipy)
```bash
pip install --upgrade scipy
```

### OpenCV ImportError
```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

### Low FPS
1. Use smaller model: `yolov8n-pose.pt` (nano)
2. Reduce frame resolution
3. Enable GPU: `device="cuda"`
4. Reduce parse_frame_skip to skip more frames

---

## Advanced Configuration

### Custom Garment Metadata

Override auto-detection ratios in `engine/utils.py`:

```python
GarmentMeta(
    path="assets/shirts/custom.png",
    name="custom",
    collar_y_ratio=0.06,    # Collar starts at 6% from top
    shoulder_y_ratio=0.14,  # Shoulders at 14% from top
    sleeve_end_ratio=0.52,  # Sleeve ends at 52% height
    hem_y_ratio=0.97,       # Hem at 97% (near bottom)
    neck_x_ratio=0.5,       # Neck centered at 50%
)
```

### Add Human Parsing Model (SCHP)

Download SCHP ATR ONNX model and place in `models/`:

```python
pipeline = RenderPipeline(
    parsing_model="models/schp_atr.onnx",
    ...
)
```

### Physics Tuning

In `engine/hybrid_warper.py`:

```python
HybridWarper(
    smooth_alpha=0.35,   # Lower = more stable, higher = more responsive
    physics_lag=0.2,     # Higher = more cloth momentum/drag
    tps_smooth=0.5,      # Higher = smoother TPS, lower = more deformation
)
```

---

## Project Structure

```
virtual_tryon/
├── main.py                     # Entry point
├── requirements.txt
├── README.md
├── assets/
│   └── shirts/                 # Add shirt PNGs here
├── models/                     # Optional: parsing/DensePose models
├── screenshots/                # Auto-created for captured frames
├── engine/
│   ├── __init__.py
│   ├── utils.py                # Shared data structures + utilities
│   ├── yolo_pose.py            # YOLOv8 pose detection
│   ├── densepose_engine.py     # Body surface estimation
│   ├── parsing_engine.py       # Human body segmentation
│   ├── garment_landmarks.py    # Auto-detect shirt control points
│   ├── hybrid_warper.py        # TPS cloth warping core
│   ├── occlusion_engine.py     # Natural body-over-shirt layering
│   ├── shadow_engine.py        # Dynamic shadows + lighting
│   └── render_pipeline.py      # Master orchestrator
└── ui/
    ├── __init__.py
    └── app.py                  # CustomTkinter GUI
```

---

## License

MIT License — Free for personal and commercial use.

---

*Built with YOLOv8 · TPS Warping · OpenCV · CustomTkinter*
