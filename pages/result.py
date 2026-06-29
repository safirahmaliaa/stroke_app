"""
pages/result.py
===============
Halaman dashboard hasil analisis lengkap.
"""

import numpy as np
import streamlit as st

from utils.session import init_session_state, get_state, set_state, clear_all
from utils.nifti_utils import find_best_axial_slice
from pipeline.overlay import create_side_by_side
from config import SEVERITY_LABELS


def _confidence_bar(prob: float, label: str):
    """Tampilkan confidence bar dengan warna."""
    pct     = int(prob * 100)
    color   = "#dc3545" if label == "Lesi" else "#28a745"
    st.markdown(f"""
    <div style="margin:0.5rem 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#9E9E9E; font-size:0.85rem;">Probabilitas Lesi</span>
            <span style="color:{color}; font-weight:700; font-size:0.95rem;">{pct}%</span>
        </div>
        <div style="background:#2D2D2D; border-radius:8px; height:12px; overflow:hidden;">
            <div style="background:{color}; width:{pct}%; height:100%;
                        border-radius:8px; transition:width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _severity_card(sev_key: str, vol_ml: float):
    """Tampilkan kartu tingkat keparahan."""
    info  = SEVERITY_LABELS[sev_key]
    color = info["color"]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1E1E2E,#2A2A3E);
                border:2px solid {color}; border-radius:14px;
                padding:1.4rem; text-align:center;">
        <div style="font-size:2.5rem;">{info['emoji']}</div>
        <div style="font-size:1.8rem; font-weight:800; color:{color};
                    margin:0.3rem 0;">{info['label'].upper()}</div>
        <div style="font-size:1.1rem; color:#E0E0E0; margin-bottom:0.5rem;">
            Volume Lesi: <strong>{vol_ml:.2f} mL</strong>
        </div>
        <div style="font-size:0.85rem; color:#9E9E9E; line-height:1.5;">
            {info['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render():
    init_session_state()

    # ── Back navigation ───────────────────────────────────────────────────────
    col_back, col_new = st.columns([1, 1])
    with col_back:
        if st.button("← Kembali ke Analisis"):
            set_state("current_page", "analysis")
            st.rerun()
    with col_new:
        if st.button("🔄  Analisis Baru", type="primary"):
            clear_all()
            set_state("current_page", "home")
            st.rerun()

    st.markdown("## 📊 Dashboard Hasil Analisis")
    st.divider()

    # ── Ambil data dari session state ─────────────────────────────────────────
    mode         = get_state("analysis_mode")
    clf_label    = get_state("clf_label")
    clf_prob     = get_state("clf_prob")
    clf_done     = get_state("clf_done")
    seg_done     = get_state("seg_done")
    vol_ml       = get_state("lesion_volume_ml")
    sev_label    = get_state("severity_label")
    overlay_frs  = get_state("overlay_frames")
    dwi_display  = get_state("dwi_display")
    image_array  = get_state("image_array")

    if not clf_done:
        st.warning("Belum ada hasil analisis. Silakan jalankan analisis terlebih dahulu.")
        return

    # ── 1. Ringkasan Klasifikasi ──────────────────────────────────────────────
    st.markdown("### 1. Hasil Klasifikasi")

    col_res, col_conf = st.columns([1, 2])

    with col_res:
        is_lesi = clf_label == "Lesi"
        badge_color = "#dc3545" if is_lesi else "#28a745"
        badge_icon  = "🔴" if is_lesi else "🟢"
        st.markdown(f"""
        <div style="background:{badge_color}22; border:2px solid {badge_color};
                    border-radius:12px; padding:1.2rem; text-align:center;">
            <div style="font-size:2.5rem;">{badge_icon}</div>
            <div style="font-size:1.6rem; font-weight:800; color:{badge_color};
                        margin-top:0.3rem;">{clf_label.upper()}</div>
            <div style="font-size:0.8rem; color:#9E9E9E; margin-top:0.2rem;">
                {'Terdeteksi lesi stroke' if is_lesi else 'Tidak terdeteksi lesi'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        if clf_prob is not None:
            _confidence_bar(clf_prob, clf_label)
            st.markdown(f"""
            <div style="background:#1E1E2E; border-radius:8px; padding:0.8rem;
                        margin-top:0.5rem; font-size:0.85rem; color:#9E9E9E;">
                <strong style="color:#E0E0E0;">Model:</strong> DenseNet121 + CBAM (M3)<br>
                <strong style="color:#E0E0E0;">Input:</strong> DWI-only, Important Slices<br>
                <strong style="color:#E0E0E0;">Threshold:</strong> 0.5
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── 2. Preview Citra Input ────────────────────────────────────────────────
    st.markdown("### 2. Citra Input")

    if mode == "quick" and image_array is not None:
        col_img, _ = st.columns([1, 2])
        with col_img:
            st.image(image_array, caption="Gambar input (Quick Analysis)", use_container_width=True)

    elif mode == "clinical" and dwi_display is not None:
        n_slices = dwi_display.shape[2]
        z_sel    = st.slider(
            "Pilih Slice Axial (DWI)",
            min_value=0,
            max_value=n_slices - 1,
            value=find_best_axial_slice(dwi_display, "brightness"),
            key="slice_slider",
        )

        cols_view = st.columns(3)
        display_items = [
            ("dwi_display",   "DWI"),
            ("adc_display",   "ADC"),
            ("flair_display", "FLAIR"),
        ]

        for i, (key, label) in enumerate(display_items):
            vol = get_state(key)
            if vol is not None:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import io
                from PIL import Image

                sl = vol[:, :, min(z_sel, vol.shape[2] - 1)].astype(np.float32)
                nz = sl[sl > 0]
                if len(nz) > 0:
                    p2, p98 = np.percentile(nz, [2, 98])
                    sl = np.clip((sl - p2) / (p98 - p2 + 1e-8), 0, 1)

                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor("#1E1E2E")
                ax.set_facecolor("#1E1E2E")
                ax.imshow(sl, cmap="gray")
                ax.axis("off")
                ax.set_title(f"{label} (z={z_sel})", color="white", fontsize=9, pad=4)
                plt.tight_layout(pad=0.2)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)
                buf.seek(0)
                cols_view[i].image(Image.open(buf), use_container_width=True)

    # ── 3. Segmentasi dan Overlay ─────────────────────────────────────────────
    if seg_done and overlay_frs:
        st.divider()
        st.markdown("### 3. Segmentasi Lesi — Overlay")
        st.markdown(
            "Warna **merah** menunjukkan area lesi yang tersegmentasi oleh NVAutoNet."
        )

        n_frames = len(overlay_frs)
        cols_ov  = st.columns(min(n_frames, 5))
        for i, frame in enumerate(overlay_frs[:5]):
            cols_ov[i].image(frame, use_container_width=True,
                             caption=f"Slice {i+1}")

        # Side-by-side dari best slice
        if dwi_display is not None:
            seg_array = get_state("seg_mask_array")
            best_z    = get_state("best_slice_idx") or find_best_axial_slice(dwi_display)
            best_z    = min(best_z, dwi_display.shape[2] - 1)

            if seg_array is not None:
                sb_img = create_side_by_side(
                    dwi_display[:, :, best_z],
                    seg_array[:, :, min(best_z, seg_array.shape[2] - 1)],
                    title=f"Slice Terbaik (z={best_z})",
                )
                st.image(sb_img, use_container_width=True)

    # ── 4. Volume dan Keparahan ───────────────────────────────────────────────
    if seg_done and vol_ml is not None and sev_label is not None:
        st.divider()
        st.markdown("### 4. Estimasi Volume dan Tingkat Keparahan")

        col_sev, col_info = st.columns([1, 2])

        with col_sev:
            _severity_card(sev_label, vol_ml)

        with col_info:
            st.markdown("**Interpretasi Klinis**")
            st.markdown(f"""
            <div style="background:#1E1E2E; border-radius:10px; padding:1rem;
                        font-size:0.88rem; line-height:1.8; color:#B0BEC5;">
                <table style="width:100%; border-collapse:collapse;">
                    <tr>
                        <td style="padding:4px 8px; color:#9E9E9E;">Volume Lesi</td>
                        <td style="padding:4px 8px; font-weight:700;
                                   color:#E0E0E0;">{vol_ml:.2f} mL</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px; color:#9E9E9E;">Kategori</td>
                        <td style="padding:4px 8px; font-weight:700;
                                   color:{SEVERITY_LABELS[sev_label]['color']};">
                            {SEVERITY_LABELS[sev_label]['label'].upper()}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px; color:#9E9E9E;">Threshold</td>
                        <td style="padding:4px 8px; color:#E0E0E0;">
                            Ringan &lt;10mL | Sedang 10–70mL | Berat &gt;70mL
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px; color:#9E9E9E;">Segmentasi</td>
                        <td style="padding:4px 8px; color:#E0E0E0;">
                            NVAutoNet (DeepISLES) — Mean DSC 0.8347
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ── 5. Ringkasan Diagnosis ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 5. Ringkasan Hasil")

    lines = []
    lines.append(f"**Hasil Klasifikasi:** {clf_label} (probabilitas: {clf_prob*100:.1f}%)")

    if seg_done and vol_ml is not None:
        sev_info = SEVERITY_LABELS[sev_label]
        lines.append(f"**Volume Lesi:** {vol_ml:.2f} mL")
        lines.append(f"**Tingkat Keparahan:** {sev_info['label']} — {sev_info['description']}")
    elif clf_label == "Lesi" and mode == "clinical":
        lines.append("**Segmentasi:** Tidak berhasil dijalankan (cek ADC file dan GPU).")
    elif mode == "quick":
        lines.append(
            "**Catatan:** Analisis volumetrik memerlukan data MRI format NIfTI (mode Clinical MRI)."
        )

    for line in lines:
        st.markdown(f"- {line}")

    st.markdown("""
    <div style="background:#1A1A2E; border-left:3px solid #FF9800; border-radius:0 8px 8px 0;
                padding:0.8rem 1rem; margin-top:1rem; font-size:0.82rem; color:#9E9E9E;">
        ⚠️ <strong style="color:#FF9800;">Disclaimer:</strong>
        Hasil analisis ini merupakan output sistem Computer-Aided Diagnosis (CAD)
        dan bukan merupakan keputusan medis final. Interpretasi klinis tetap menjadi
        wewenang dokter atau tenaga medis yang berkompeten.
    </div>
    """, unsafe_allow_html=True)