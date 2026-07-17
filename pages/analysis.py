"""
pages/analysis.py
=================
Flow:
  Fase 1 (di sini) : Preprocessing + Klasifikasi → langsung ke result
  Fase 2 (di result): Segmentasi + Volume + Overlay (jalan otomatis via seg_pending)
"""
import os
import time
import numpy as np
import streamlit as st

from utils.session import init_session_state, get_state, set_state, clear_results
from utils.nifti_utils import load_nifti, transpose_to_axial, find_best_axial_slice
from utils.image_utils import load_image_2d, prepare_image_tensor
from pipeline.preprocessing import prepare_clinical_volumes
from pipeline.classifier import predict_slices, predict_single_image
from pipeline.segmenter import run_segmentation
from pipeline.volume import calculate_volume_ml
from pipeline.severity import estimate_severity
from pipeline.overlay import create_overlay_gallery
from config import UPLOADS_DIR, OUTPUTS_DIR


def _save_upload(uploaded_file, suffix: str = "") -> str:
    dest = UPLOADS_DIR / f"{uploaded_file.name}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def _preview_nifti_slice(vol_axial: np.ndarray, label: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, io
    from PIL import Image

    z  = find_best_axial_slice(vol_axial, mode="brightness")
    sl = vol_axial[:, :, z].astype(np.float32)
    nz = sl[sl > 0]
    if len(nz) > 0:
        p2, p98 = np.percentile(nz, [2, 98])
        sl = np.clip((sl - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("#1E1E2E"); ax.set_facecolor("#1E1E2E")
    ax.imshow(sl, cmap="gray"); ax.axis("off")
    ax.set_title(label, color="white", fontsize=9, pad=4)
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return Image.open(buf)


# ── FASE 1: Klasifikasi saja ──────────────────────────────────────────────────

def _run_classification_phase(dwi_path, adc_path, flair_path) -> bool:
    prog = st.progress(0, text="Memuat volume MRI...")

    dwi_vol   = load_nifti(dwi_path)
    adc_vol   = load_nifti(adc_path)   if adc_path   else None
    flair_vol = load_nifti(flair_path) if flair_path else None

    if dwi_vol is None:
        st.error("Gagal membaca file DWI. Pastikan format NIfTI valid.")
        return False

    prog.progress(20, text="Preprocessing slice...")
    dwi_axial, adc_axial, flair_axial, slices = prepare_clinical_volumes(
        dwi_vol, adc_vol, flair_vol
    )
    if not slices:
        st.error("Tidak ada slice valid pada file DWI.")
        return False

    set_state("dwi_display",   dwi_axial)
    set_state("adc_display",   adc_axial)
    set_state("flair_display", flair_axial)

    prog.progress(50, text="Menjalankan klasifikasi DenseNet121+CBAM...")
    clf = predict_slices(slices)
    set_state("clf_label", clf["label"])
    set_state("clf_prob",  clf["probability"])
    set_state("clf_done",  True)

    prog.progress(100, text=f"✅ Klasifikasi selesai: {clf['label']}")
    time.sleep(0.4)

    # Tandai apakah segmentasi perlu dilanjutkan di result
    set_state("seg_pending", clf["label"] == "Lesi" and adc_path is not None)
    return True


# ── FASE 2: Segmentasi — dipanggil dari result.py ─────────────────────────────

def run_segmentation_phase():
    """Dipanggil dari pages/result.py saat seg_pending=True."""
    dwi_path = get_state("dwi_path")
    adc_path = get_state("adc_path")

    st.write("🔄 Menjalankan NVAutoNet ensemble (15 checkpoint)...")
    out_mask = str(OUTPUTS_DIR / "lesion_mask.nii.gz")
    seg      = run_segmentation(adc_path, dwi_path, out_mask)

    set_state("seg_pending", False)

    if not seg["success"]:
        st.warning(f"Segmentasi gagal: {seg['error']}")
        return

    if seg.get("error"):
        st.warning(f"Peringatan: {seg['error']}")

    set_state("seg_mask_path", seg["mask_path"])
    set_state("seg_done",      True)

    st.write("🔄 Menghitung volume lesi...")
    vol_ml = calculate_volume_ml(seg["mask_sitk"])
    set_state("lesion_volume_ml", vol_ml)

    sev = estimate_severity(vol_ml)
    set_state("severity_label", sev["key"])
    set_state("severity_done",  True)

    st.write("🔄 Membuat visualisasi overlay...")
    try:
        grid_img, best_z = create_overlay_gallery(
            dwi_path=dwi_path,
            mask_path=seg["mask_path"],
            vol_ml=vol_ml,
            subject_id="MRI Pasien",
        )
        set_state("overlay_grid",   grid_img)
        set_state("best_slice_idx", best_z)
    except Exception as e:
        st.warning(f"Overlay gagal: {e}")

    st.write("✅ Segmentasi selesai!")


# ── Quick Analysis ────────────────────────────────────────────────────────────

def _run_quick_pipeline(image_array) -> bool:
    prog = st.progress(0, text="Menyiapkan gambar...")
    set_state("image_array", image_array)
    prog.progress(40, text="Menjalankan klasifikasi DenseNet121+CBAM...")
    clf = predict_single_image(prepare_image_tensor(image_array))
    set_state("clf_label", clf["label"])
    set_state("clf_prob",  clf["probability"])
    set_state("clf_done",  True)
    set_state("seg_pending", False)
    prog.progress(100, text="✅ Klasifikasi selesai!")
    time.sleep(0.3)
    return True


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    init_session_state()
    mode = get_state("analysis_mode")

    if st.button("← Kembali ke Home"):
        set_state("current_page", "home"); st.rerun()

    if mode == "clinical":   _render_clinical()
    elif mode == "quick":    _render_quick()
    else:
        st.warning("Pilih mode analisis dari halaman Home.")
        if st.button("Ke Home"):
            set_state("current_page", "home"); st.rerun()


def _render_clinical():
    st.markdown("## 🏥 Clinical MRI Analysis")
    st.markdown("Upload file MRI NIfTI. DWI wajib. ADC diperlukan untuk segmentasi.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**DWI** *(wajib)*")
        dwi_file = st.file_uploader("Upload DWI", type=["nii", "gz"],
                                    key="up_dwi", label_visibility="collapsed")
        if dwi_file: st.success(f"✅ {dwi_file.name}")
    with col2:
        st.markdown("**ADC** *(wajib untuk segmentasi)*")
        adc_file = st.file_uploader("Upload ADC", type=["nii", "gz"],
                                    key="up_adc", label_visibility="collapsed")
        if adc_file: st.success(f"✅ {adc_file.name}")
    with col3:
        st.markdown("**FLAIR** *(opsional)*")
        flair_file = st.file_uploader("Upload FLAIR", type=["nii", "gz"],
                                      key="up_flair", label_visibility="collapsed")
        if flair_file: st.success(f"✅ {flair_file.name}")

    if dwi_file:
        st.divider()
        st.markdown("##### Preview Slice Tengah")
        prev_cols = st.columns(3)
        with st.spinner("Memuat preview..."):
            try:
                dwi_vol_tmp = load_nifti(_save_upload(dwi_file))
                if dwi_vol_tmp is not None:
                    prev_cols[0].image(
                        _preview_nifti_slice(transpose_to_axial(dwi_vol_tmp), "DWI"),
                        use_container_width=True)
                if adc_file:
                    adc_vol_tmp = load_nifti(_save_upload(adc_file))
                    if adc_vol_tmp is not None:
                        prev_cols[1].image(
                            _preview_nifti_slice(transpose_to_axial(adc_vol_tmp), "ADC"),
                            use_container_width=True)
                if flair_file:
                    flair_vol_tmp = load_nifti(_save_upload(flair_file))
                    if flair_vol_tmp is not None:
                        prev_cols[2].image(
                            _preview_nifti_slice(transpose_to_axial(flair_vol_tmp), "FLAIR"),
                            use_container_width=True)
            except Exception as e:
                st.warning(f"Preview gagal: {e}")

    st.divider()
    if not dwi_file:
        st.info("Upload minimal file DWI untuk memulai analisis.")

    if st.button("🔍  Jalankan Analisis", disabled=dwi_file is None,
                 use_container_width=True, type="primary"):
        clear_results()
        set_state("pipeline_running", True)
        try:
            dwi_path   = _save_upload(dwi_file)
            adc_path   = _save_upload(adc_file)   if adc_file   else None
            flair_path = _save_upload(flair_file) if flair_file else None
            set_state("dwi_path",   dwi_path)
            set_state("adc_path",   adc_path)
            set_state("flair_path", flair_path)

            if _run_classification_phase(dwi_path, adc_path, flair_path):
                set_state("pipeline_running", False)
                set_state("current_page", "result")
                st.rerun()
        except Exception as e:
            set_state("pipeline_running", False)
            set_state("pipeline_error", str(e))
            st.error(f"Pipeline error: {e}")


def _render_quick():
    st.markdown("## ⚡ Quick Analysis")
    st.markdown("Upload gambar MRI otak (JPG/PNG) untuk klasifikasi biner.")
    st.info("Mode ini hanya menghasilkan klasifikasi. "
            "Untuk analisis volumetrik gunakan Clinical MRI.", icon="ℹ️")
    st.divider()

    img_file = st.file_uploader("Upload gambar MRI", type=["jpg", "jpeg", "png"], key="up_img")
    if img_file:
        col_prev, col_info = st.columns([1, 2])
        image_array = load_image_2d(img_file)
        with col_prev:
            st.image(image_array, caption="Preview", use_container_width=True)
        with col_info:
            st.markdown(f"**Nama file:** `{img_file.name}`")
            st.markdown(f"**Ukuran:** {img_file.size / 1024:.1f} KB")
            st.markdown(f"**Dimensi:** {image_array.shape[1]} × {image_array.shape[0]} px")
        st.divider()
        if st.button("🔍  Jalankan Klasifikasi", use_container_width=True, type="primary"):
            clear_results()
            set_state("pipeline_running", True)
            try:
                if _run_quick_pipeline(image_array):
                    set_state("pipeline_running", False)
                    set_state("current_page", "result")
                    st.rerun()
            except Exception as e:
                set_state("pipeline_running", False)
                set_state("pipeline_error", str(e))
                st.error(f"Error: {e}")
EOF
echo "analysis.py OK"
