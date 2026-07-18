"""
config.py — semua konstanta dan path aplikasi StrokeVision AI
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# ── Weights — path relatif dari root project ──────────────────────────────────
CLASSIFIER_WEIGHTS = BASE_DIR / "weights" / "classifier" / "M3_DWI_ImportantSlice_best.pth"
NVAUTONET_WEIGHTS = Path("/content/drive/MyDrive/DeepIsles/weights/NVAUTO/ts")
NVAUTONET_N_CKPT   = 15

# ── Direktori I/O ─────────────────────────────────────────────────────────────
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploaded_files"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Preprocessing (notebook Cell 4) ──────────────────────────────────────────
IMG_SIZE          = 224
CLAHE_CLIP_LIMIT  = 0.02
GAUSSIAN_SIGMA    = 0.5
BRAIN_COV_THRESH  = 0.05
LESI_THRESH       = 0.001
OUTLIER_LOW_PCT   = 1
OUTLIER_HIGH_PCT  = 99
MAX_LESI_SLICES   = 10
MAX_NOLESI_SLICES = 8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Klasifikasi ───────────────────────────────────────────────────────────────
DROPOUT            = 0.35
CBAM_REDUCTION     = 16
CBAM_KERNEL        = 7
CLASSIFY_THRESHOLD = 0.5

# ── Segmentasi NVAutoNet (notebook nvauto Cell 7) ─────────────────────────────
NVAUTO_ROI_SIZE   = [192, 192, 128]
NVAUTO_OVERLAP    = 0.625
NVAUTO_SW_BATCH   = 2
NVAUTO_SPACING    = [1.0, 1.0, 1.0]
NVAUTO_SEG_THRESH = 0.5

# ── Keparahan (notebook nvauto Cell 10) ──────────────────────────────────────
SEVERITY_RINGAN_MAX = 10.0
SEVERITY_SEDANG_MAX = 70.0

SEVERITY_LABELS = {
    "RINGAN": {
        "label": "Ringan", "color": "#28a745",
        "description": "Volume lesi < 10 mL. Defisit neurologis minimal.",
        "emoji": "🟢",
    },
    "SEDANG": {
        "label": "Sedang", "color": "#fd7e14",
        "description": "Volume lesi 10–70 mL. Perlu penanganan segera.",
        "emoji": "🟡",
    },
    "BERAT": {
        "label": "Berat", "color": "#dc3545",
        "description": "Volume lesi > 70 mL. Risiko tinggi, penanganan intensif.",
        "emoji": "🔴",
    },
}

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE        = "StrokeVision AI"
APP_SUBTITLE     = "Sistem Deteksi dan Estimasi Keparahan Stroke Iskemik"
APP_ICON         = "🧠"
ALLOWED_NIFTI_TYPES = [".nii", ".nii.gz"]
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SEED = 42
