"""
config.py
=========
Seluruh konstanta, path, dan hyperparameter aplikasi stroke detection.
Semua nilai diambil langsung dari notebook penelitian (source of truth).
"""

import os
from pathlib import Path

# ── Root Project ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Weights ───────────────────────────────────────────────────────────────────
CLASSIFIER_WEIGHTS = Path(r"G:\My Drive\SKRIPSI_FINAL_20260510_1708\03_MODEL_M3_DWI_ImportantSlice\M3_DWI_ImportantSlice_best.pth")
NVAUTONET_WEIGHTS = Path(r"G:\My Drive\DeepIsles\weights\NVAUTO\ts")
NVAUTONET_N_CKPT = 15  # jumlah file model0.ts ... model14.ts

# ── Direktori I/O ─────────────────────────────────────────────────────────────
OUTPUTS_DIR      = BASE_DIR / "outputs"
UPLOADS_DIR      = BASE_DIR / "uploaded_files"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Preprocessing (sesuai notebook Cell 4) ────────────────────────────────────
IMG_SIZE          = 224          # target resize H×W
CLAHE_CLIP_LIMIT  = 0.02         # clip_limit CLAHE
GAUSSIAN_SIGMA    = 0.5          # sigma Gaussian filter
BRAIN_COV_THRESH  = 0.05         # buang slice jika brain coverage < 5%
LESI_THRESH       = 0.001        # threshold labeling lesi (0.1% piksel)
OUTLIER_LOW_PCT   = 1            # percentile bawah clip outlier
OUTLIER_HIGH_PCT  = 99           # percentile atas clip outlier

# Important Slices (M3)
MAX_LESI_SLICES   = 10           # ambil 10 slice dengan lesi proxy terbesar
MAX_NOLESI_SLICES = 8            # ambil 8 slice tengah volume

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Model Klasifikasi ─────────────────────────────────────────────────────────
DROPOUT           = 0.35         # dropout rate (sesuai notebook)
CBAM_REDUCTION    = 16           # reduction ratio Channel Attention
CBAM_KERNEL       = 7            # kernel size Spatial Attention
CLASSIFY_THRESHOLD = 0.5         # threshold sigmoid → biner

# ── Segmentasi NVAutoNet (sesuai notebook Cell 7) ─────────────────────────────
NVAUTO_ROI_SIZE    = [192, 192, 128]
NVAUTO_OVERLAP     = 0.625
NVAUTO_SW_BATCH    = 2
NVAUTO_SPACING     = [1.0, 1.0, 1.0]   # target isotropic spacing (mm)
NVAUTO_SEG_THRESH  = 0.5                # threshold probabilitas → mask biner

# ── Estimasi Keparahan (dari notebook Cell 10 visualisasi) ────────────────────
SEVERITY_RINGAN_MAX = 10.0   # < 10 mL  → RINGAN
SEVERITY_SEDANG_MAX = 70.0   # 10-70 mL → SEDANG
                              # > 70 mL  → BERAT

SEVERITY_LABELS = {
    "RINGAN": {
        "label"      : "Ringan",
        "color"      : "#28a745",
        "description": "Volume lesi < 10 mL. Defisit neurologis minimal.",
        "emoji"      : "🟢",
    },
    "SEDANG": {
        "label"      : "Sedang",
        "color"      : "#fd7e14",
        "description": "Volume lesi 10–70 mL. Perlu penanganan segera.",
        "emoji"      : "🟡",
    },
    "BERAT": {
        "label"      : "Berat",
        "color"      : "#dc3545",
        "description": "Volume lesi > 70 mL. Risiko tinggi, penanganan intensif.",
        "emoji"      : "🔴",
    },
}

# ── UI / Streamlit ────────────────────────────────────────────────────────────
APP_TITLE       = "StrokeVision AI"
APP_SUBTITLE    = "Sistem Deteksi dan Estimasi Keparahan Stroke Iskemik"
APP_DESCRIPTION = (
    "Aplikasi berbasis Deep Learning untuk klasifikasi stroke iskemik "
    "dan estimasi tingkat keparahan menggunakan citra MRI otak. "
    "Dibangun dengan DenseNet121 + CBAM untuk klasifikasi dan "
    "NVAutoNet untuk segmentasi lesi."
)
APP_ICON        = "🧠"

ALLOWED_NIFTI_TYPES = [".nii", ".nii.gz"]
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42