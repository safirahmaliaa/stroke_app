"""utils/session.py — Streamlit session state management"""
import streamlit as st
from typing import Any

SESSION_DEFAULTS = {
    "current_page"      : "home",
    "analysis_mode"     : None,
    # File paths
    "dwi_path"          : None,
    "adc_path"          : None,
    "flair_path"        : None,
    # Display arrays (axial)
    "dwi_display"       : None,
    "adc_display"       : None,
    "flair_display"     : None,
    # Quick mode
    "image_array"       : None,
    # Klasifikasi
    "clf_label"         : None,
    "clf_prob"          : None,
    "clf_done"          : False,
    # Segmentasi
    "seg_mask_path"     : None,
    "seg_done"          : False,
    "seg_pending"       : False,   # True = segmentasi belum jalan, lanjut di result
    # Volume & keparahan
    "lesion_volume_ml"  : None,
    "severity_label"    : None,
    "severity_done"     : False,
    # Overlay
    "overlay_grid"      : None,
    "best_slice_idx"    : None,
    # Pipeline flags
    "pipeline_running"  : False,
    "pipeline_error"    : None,
}

def init_session_state():
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def set_state(key: str, value: Any):
    st.session_state[key] = value

def get_state(key: str) -> Any:
    return st.session_state.get(key, SESSION_DEFAULTS.get(key))

def clear_results():
    result_keys = [
        "clf_label", "clf_prob", "clf_done",
        "seg_mask_path", "seg_done", "seg_pending",
        "lesion_volume_ml", "severity_label", "severity_done",
        "overlay_grid", "best_slice_idx",
        "pipeline_running", "pipeline_error",
    ]
    for key in result_keys:
        st.session_state[key] = SESSION_DEFAULTS[key]

def clear_all():
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = default

def is_result_ready() -> bool:
    return bool(st.session_state.get("clf_done", False))

def is_segmentation_ready() -> bool:
    return bool(st.session_state.get("seg_done", False))
