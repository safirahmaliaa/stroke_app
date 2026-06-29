"""
utils/session.py
================
Manajemen Streamlit session state.
Semua key yang digunakan lintas halaman didefinisikan di sini
agar konsisten dan mudah di-debug.
"""

import streamlit as st
from typing import Any


# ── Key Definitions ───────────────────────────────────────────────────────────
# Daftarkan semua session key beserta default value-nya di sini.
SESSION_DEFAULTS = {
    # Navigasi
    "current_page"        : "home",

    # Mode analisis yang dipilih user
    "analysis_mode"       : None,        # "clinical" | "quick"

    # File paths (untuk mode clinical)
    "dwi_path"            : None,
    "adc_path"            : None,
    "flair_path"          : None,

    # Numpy array gambar untuk display
    "dwi_display"         : None,        # (H, W, N) axial
    "adc_display"         : None,
    "flair_display"       : None,

    # Untuk mode quick
    "image_array"         : None,        # (H, W, 3) uint8

    # Hasil klasifikasi
    "clf_label"           : None,        # "Lesi" | "Non-Lesi"
    "clf_prob"            : None,        # float 0-1
    "clf_done"            : False,

    # Hasil segmentasi
    "seg_mask_path"       : None,        # path NIfTI mask hasil NVAutoNet
    "seg_mask_array"      : None,        # (H, W, N) numpy biner
    "seg_done"            : False,

    # Volume dan keparahan
    "lesion_volume_ml"    : None,        # float
    "severity_label"      : None,        # "RINGAN" | "SEDANG" | "BERAT"
    "severity_done"       : False,

    # Overlay frames untuk visualisasi
    "overlay_frames"      : None,        # list of (H, W, 3) uint8
    "best_slice_idx"      : None,        # int

    # Flag pipeline
    "pipeline_running"    : False,
    "pipeline_error"      : None,        # str pesan error jika ada
}


def init_session_state():
    """
    Inisialisasi semua session state key dengan default value.
    Dipanggil di awal setiap halaman.
    """
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def set_state(key: str, value: Any):
    """Shortcut untuk set session state."""
    st.session_state[key] = value


def get_state(key: str) -> Any:
    """Shortcut untuk get session state dengan fallback ke default."""
    return st.session_state.get(key, SESSION_DEFAULTS.get(key))


def clear_results():
    """
    Reset semua hasil analisis tanpa mereset file upload
    dan mode yang dipilih. Dipanggil saat user upload ulang.
    """
    result_keys = [
        "clf_label", "clf_prob", "clf_done",
        "seg_mask_path", "seg_mask_array", "seg_done",
        "lesion_volume_ml", "severity_label", "severity_done",
        "overlay_frames", "best_slice_idx",
        "pipeline_running", "pipeline_error",
    ]
    for key in result_keys:
        st.session_state[key] = SESSION_DEFAULTS[key]


def clear_all():
    """Reset seluruh session state ke default. Dipanggil saat mulai analisis baru."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = default


def is_result_ready() -> bool:
    """Cek apakah hasil klasifikasi sudah tersedia."""
    return bool(st.session_state.get("clf_done", False))


def is_segmentation_ready() -> bool:
    """Cek apakah hasil segmentasi sudah tersedia."""
    return bool(st.session_state.get("seg_done", False))