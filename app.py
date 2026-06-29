"""
app.py
======
Entry point utama aplikasi StrokeVision AI.
Mengelola navigasi multi-halaman menggunakan session state.

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
from utils.session import init_session_state, get_state

# ── Konfigurasi halaman (HARUS dipanggil pertama kali) ────────────────────────
st.set_page_config(
    page_title  = "StrokeVision AI",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom CSS global ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font dan background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #0D1117;
        color: #E0E0E0;
    }

    /* Sembunyikan sidebar toggle default */
    [data-testid="collapsedControl"] { display: none; }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,195,247,0.3);
    }

    /* Metric styling */
    [data-testid="metric-container"] {
        background: #1E1E2E;
        border: 1px solid #2D2D3E;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Upload area */
    [data-testid="stFileUploaderDropzone"] {
        background: #1A1A2E;
        border: 2px dashed #4FC3F7;
        border-radius: 10px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4FC3F7, #AB47BC);
    }

    /* Divider */
    hr { border-color: #2D2D3E; }

    /* Slider */
    [data-testid="stSlider"] > div > div > div > div {
        background: #4FC3F7;
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Navigasi ──────────────────────────────────────────────────────────────────
init_session_state()
page = get_state("current_page")

if page == "home":
    from pages.home import render
    render()
elif page == "analysis":
    from pages.analysis import render
    render()
elif page == "result":
    from pages.result import render
    render()
else:
    from pages.home import render
    render()