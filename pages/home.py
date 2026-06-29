"""
pages/home.py
=============
Halaman landing page StrokeVision AI.
"""

import streamlit as st
from utils.session import init_session_state, clear_all, set_state


def render():
    init_session_state()

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <div style="font-size:4rem;">🧠</div>
        <h1 style="font-size:2.6rem; font-weight:800; margin:0.3rem 0 0.2rem 0;
                   background: linear-gradient(90deg,#4FC3F7,#AB47BC);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            StrokeVision AI
        </h1>
        <p style="font-size:1.1rem; color:#9E9E9E; margin:0 0 0.4rem 0;">
            Sistem Deteksi dan Estimasi Keparahan Stroke Iskemik
        </p>
        <p style="font-size:0.85rem; color:#616161;">
            DenseNet121 + CBAM &nbsp;|&nbsp; NVAutoNet (DeepISLES) &nbsp;|&nbsp; ISLES 2022
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Deskripsi Singkat ─────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Tentang Aplikasi")
        st.markdown("""
        **StrokeVision AI** adalah sistem *Computer-Aided Diagnosis* (CAD) berbasis
        *Deep Learning* yang dirancang untuk membantu praktisi medis dalam:

        - Mengklasifikasikan keberadaan lesi stroke iskemik pada citra MRI otak
        - Mensegmentasi area lesi secara otomatis menggunakan model 3D volumetrik
        - Mengestimasi tingkat keparahan stroke berdasarkan volume lesi (mL)
        - Memvisualisasikan hasil dalam dashboard interaktif yang mudah dibaca

        Sistem ini dikembangkan sebagai bagian dari penelitian skripsi di
        **Program Studi Sains Data, UPN Veteran Jawa Timur**.
        """)

    with col2:
        st.markdown("### Performa Model")
        st.markdown("""
        **Klasifikasi (M3 — DWI Important Slices)**
        """)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy", "84.65%")
        col_b.metric("F1-Score", "0.8598")
        col_c.metric("AUC", "0.9195")

        st.markdown("**Segmentasi (NVAutoNet)**")
        st.metric("Mean Dice (DSC)", "0.8347")

    st.divider()

    # ── Workflow Sistem ───────────────────────────────────────────────────────
    st.markdown("### Alur Kerja Sistem")

    st.markdown("""
    <div style="background:#1E1E2E; border-radius:12px; padding:1.5rem; margin-bottom:1rem;">
    <table style="width:100%; border-collapse:collapse; text-align:center; color:#E0E0E0;">
    <tr>
        <td style="padding:0.6rem;">
            <div style="font-size:1.8rem;">📤</div>
            <div style="font-size:0.8rem; color:#9E9E9E;">Upload MRI</div>
        </td>
        <td style="color:#4FC3F7; font-size:1.4rem;">→</td>
        <td style="padding:0.6rem;">
            <div style="font-size:1.8rem;">⚙️</div>
            <div style="font-size:0.8rem; color:#9E9E9E;">Preprocessing</div>
        </td>
        <td style="color:#4FC3F7; font-size:1.4rem;">→</td>
        <td style="padding:0.6rem;">
            <div style="font-size:1.8rem;">🤖</div>
            <div style="font-size:0.8rem; color:#9E9E9E;">Klasifikasi</div>
        </td>
        <td style="color:#4FC3F7; font-size:1.4rem;">→</td>
        <td style="padding:0.6rem;">
            <div style="font-size:1.8rem;">🔬</div>
            <div style="font-size:0.8rem; color:#9E9E9E;">Segmentasi</div>
        </td>
        <td style="color:#4FC3F7; font-size:1.4rem;">→</td>
        <td style="padding:0.6rem;">
            <div style="font-size:1.8rem;">📊</div>
            <div style="font-size:0.8rem; color:#9E9E9E;">Dashboard Hasil</div>
        </td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Dua Mode Analisis ─────────────────────────────────────────────────────
    st.markdown("### Pilih Mode Analisis")

    col_m1, col_m2 = st.columns(2, gap="large")

    with col_m1:
        st.markdown("""
        <div style="background:#1A2744; border:1px solid #4FC3F7; border-radius:12px;
                    padding:1.4rem; min-height:220px;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">🏥</div>
            <h4 style="color:#4FC3F7; margin:0 0 0.6rem 0;">Clinical MRI</h4>
            <p style="color:#B0BEC5; font-size:0.9rem; line-height:1.6;">
                Mode utama untuk data klinis. Upload file MRI dalam format
                NIfTI (.nii / .nii.gz).
            </p>
            <ul style="color:#9E9E9E; font-size:0.85rem; padding-left:1.2rem;">
                <li>DWI — <em>wajib</em></li>
                <li>ADC — wajib (untuk segmentasi)</li>
                <li>FLAIR — opsional</li>
            </ul>
            <p style="color:#81C784; font-size:0.82rem; margin-top:0.8rem;">
                ✅ Klasifikasi + Segmentasi + Volume Lesi + Keparahan
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🏥  Mulai Clinical MRI", use_container_width=True, type="primary"):
            clear_all()
            set_state("analysis_mode", "clinical")
            set_state("current_page", "analysis")
            st.rerun()

    with col_m2:
        st.markdown("""
        <div style="background:#1F2A1A; border:1px solid #81C784; border-radius:12px;
                    padding:1.4rem; min-height:220px;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">⚡</div>
            <h4 style="color:#81C784; margin:0 0 0.6rem 0;">Quick Analysis</h4>
            <p style="color:#B0BEC5; font-size:0.9rem; line-height:1.6;">
                Mode cepat untuk screening awal. Upload gambar 2D MRI
                dalam format umum.
            </p>
            <ul style="color:#9E9E9E; font-size:0.85rem; padding-left:1.2rem;">
                <li>JPG / JPEG</li>
                <li>PNG</li>
            </ul>
            <p style="color:#FFB74D; font-size:0.82rem; margin-top:0.8rem;">
                ⚡ Klasifikasi saja (analisis volumetrik butuh NIfTI)
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("⚡  Mulai Quick Analysis", use_container_width=True):
            clear_all()
            set_state("analysis_mode", "quick")
            set_state("current_page", "analysis")
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#424242; font-size:0.78rem; padding:0.5rem 0;">
        Safira Rahmalia Putri · NPM 22083010073 · Sains Data UPN Veteran Jawa Timur · 2026
        <br>Pembimbing: Dr. I Gede Susrama Mas Diyasa, ST., MT. &nbsp;|&nbsp; Andri Fauzan Adziima, M.Si.
        <br><br>⚠️ Aplikasi ini merupakan alat bantu diagnosis, bukan pengganti keputusan klinis dokter.
    </div>
    """, unsafe_allow_html=True)