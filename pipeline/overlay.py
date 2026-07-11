"""
pipeline/overlay.py
===================
Visualisasi overlay lesion mask di atas citra DWI.

Dikonversi IDENTIK dari notebook visualisasi_overlay_lengkap.ipynb
(fungsi normalize_for_display, red_overlay, load_and_sync_masks, plot_case).

ATURAN KOORDINAT — KRITIS:
  Notebook memuat DWI via nibabel.get_fdata() tanpa transpose apapun → shape (X, Y, Z).
  Mask diload via sitk lalu .transpose(2,1,0) → shape (X, Y, Z).
  Transpose .T hanya dilakukan PER SLICE di dalam plot_case:
      dwi_sl  = dwi[:, :, sl].T
      pred_sl = pred_biner[:, :, sl].T
  Seluruh fungsi ini mengikuti konvensi yang sama.
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Tuple, Optional


# ── Fungsi inti — IDENTIK notebook Cell 4 ────────────────────────────────────

def normalize_for_display(arr_2d: np.ndarray) -> np.ndarray:
    """Normalisasi slice 2D untuk display (exclude background 0)."""
    brain = arr_2d > 0
    if brain.sum() == 0:
        return np.zeros_like(arr_2d, dtype=np.float32)
    p2, p98 = np.percentile(arr_2d[brain], [2, 98])
    out = np.clip((arr_2d - p2) / (p98 - p2 + 1e-8), 0, 1)
    out[~brain] = 0
    return out.astype(np.float32)


def red_overlay(disp: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Buat RGBA image dengan overlay merah di area lesi."""
    h, w = disp.shape
    rgb  = np.stack([disp, disp, disp], axis=-1)
    rgba = np.concatenate([rgb, np.ones((h, w, 1))], axis=-1).astype(np.float32)
    if mask_2d.sum() > 0:
        ov = np.zeros((h, w, 4), dtype=np.float32)
        ov[mask_2d == 1] = [1.0, 0.0, 0.0, alpha]
        a = ov[..., 3:4]
        rgba[..., :3] = rgba[..., :3] * (1 - a) + ov[..., :3] * a
    return np.clip(rgba, 0, 1)


def load_and_sync_masks(dwi_path: str, pred_path: str, gt_path: str = None) -> tuple:
    """
    Load DWI + pred mask, sinkronkan via SimpleITK physical space resampling.
    IDENTIK dengan notebook load_and_sync_masks().

    Returns
    -------
    dwi        : (X, Y, Z) float32 — nibabel raw, TANPA transpose
    pred_biner : (X, Y, Z) uint8  — sitk array setelah .transpose(2,1,0)
    gt_biner   : (X, Y, Z) uint8 atau None
    vol_ml     : float
    """
    # Load DWI via nibabel — TANPA canonical transform, TANPA transpose
    dwi_nib = nib.load(dwi_path)
    dwi     = dwi_nib.get_fdata().astype(np.float32)
    if dwi.ndim == 4:
        dwi = dwi[..., -1]

    # Load pred mask via sitk
    pred_sitk = sitk.ReadImage(pred_path)
    gt_sitk   = sitk.ReadImage(dwi_path)
    if gt_sitk.GetDimension() == 4:
        gt_sitk = sitk.Extract(
            gt_sitk, list(gt_sitk.GetSize()[:3]) + [0], [0, 0, 0, 0]
        )

    # Resample pred ke DWI space (physical space resampling)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(gt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    pred_resampled = resampler.Execute(pred_sitk)

    # sitk GetArrayFromImage → (Z,Y,X), transpose ke (X,Y,Z)
    pred_arr   = sitk.GetArrayFromImage(pred_resampled)      # (Z, Y, X)
    pred_biner = (pred_arr.transpose(2, 1, 0) > 0).astype(np.uint8)  # (X, Y, Z)

    # Volume dari pred asli (sebelum resample)
    sp     = pred_sitk.GetSpacing()
    vol_ml = float(
        (sitk.GetArrayFromImage(pred_sitk) > 0).sum()
    ) * sp[0] * sp[1] * sp[2] / 1000

    # Load GT jika ada
    gt_biner = None
    if gt_path and os.path.exists(gt_path):
        gt_sitk2 = sitk.ReadImage(gt_path)
        res2 = sitk.ResampleImageFilter()
        res2.SetReferenceImage(gt_sitk)
        res2.SetInterpolator(sitk.sitkNearestNeighbor)
        res2.SetDefaultPixelValue(0)
        gt_arr   = sitk.GetArrayFromImage(res2.Execute(gt_sitk2))
        gt_biner = (gt_arr.transpose(2, 1, 0) > 0).astype(np.uint8)

    return dwi, pred_biner, gt_biner, float(vol_ml)


def plot_case(
    dwi: np.ndarray,
    pred_biner: np.ndarray,
    vol_ml: float,
    subject_id: str = "",
    slices: Optional[List[int]] = None,
    n_slices: int = 5,
    gt_biner: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Plot grid overlay — IDENTIK dengan notebook plot_case().

    dwi dan pred_biner shape (X, Y, Z) — belum di-transpose.
    Transpose .T dilakukan per-slice di dalam fungsi ini.

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    n_sl    = pred_biner.shape[2]
    best_sl = int(np.argmax([pred_biner[:, :, z].sum() for z in range(n_sl)]))

    if slices is None:
        offsets = np.linspace(-8, 8, n_slices, dtype=int)
        slices  = sorted(set([max(0, min(n_sl - 1, best_sl + o)) for o in offsets]))

    n_rows = 3 if gt_biner is not None else 2

    fig, axes = plt.subplots(
        n_rows, len(slices),
        figsize=(3.5 * len(slices), 3.2 * n_rows + 1.2)
    )
    fig.patch.set_facecolor('#0D1117')

    if len(slices) == 1:
        axes = axes.reshape(n_rows, 1)

    for col, sl in enumerate(slices):
        sl = max(0, min(n_sl - 1, sl))

        # KRITIS: .T persis notebook — transpose di sini, bukan sebelumnya
        dwi_sl  = dwi[:, :, sl].T
        pred_sl = pred_biner[:, :, sl].T
        disp    = normalize_for_display(dwi_sl)

        for row in range(n_rows):
            axes[row, col].set_facecolor('#0D1117')
            axes[row, col].axis('off')

        # Row 0: DWI grayscale
        axes[0, col].imshow(disp, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(
            f'sl.{sl}' + (' ←' if sl == best_sl else ''),
            color='#58A6FF' if sl == best_sl else '#8B949E',
            fontsize=9,
            fontweight='bold' if sl == best_sl else 'normal',
        )

        if gt_biner is not None:
            gt_sl = gt_biner[:, :, sl].T
            axes[1, col].imshow(red_overlay(disp, gt_sl, 0.60))   # hijau diganti merah juga untuk demo
            if col == 0:
                axes[1, col].set_ylabel('Ground Truth', color='#7EC8A0', fontsize=10, labelpad=8)
            axes[2, col].imshow(red_overlay(disp, pred_sl, 0.65))
            if col == 0:
                axes[2, col].set_ylabel('Lesion Overlay', color='#FF7B7B', fontsize=10, labelpad=8)
        else:
            # Row 1: Pred merah — tidak ada GT
            axes[1, col].imshow(red_overlay(disp, pred_sl, 0.65))
            if col == 0:
                axes[1, col].set_ylabel('Lesion Overlay', color='#FF7B7B', fontsize=10, labelpad=8)

    sev = 'RINGAN' if vol_ml < 10 else ('SEDANG' if vol_ml < 70 else 'BERAT')

    plt.suptitle(
        f'{subject_id}\nVol: {vol_ml:.2f} mL  |  {sev}',
        color='white', fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0D1117')
    plt.close(fig)
    buf.seek(0)
    result = np.array(Image.open(buf).convert('RGB'))
    buf.close()
    return result


# ── Entry point untuk Streamlit ───────────────────────────────────────────────

def create_overlay_gallery(
    dwi_path: str,
    mask_path: str,
    vol_ml: float,
    subject_id: str = "MRI",
    n_display: int = 5,
) -> Tuple[np.ndarray, int]:
    """
    Entry point dari pipeline Streamlit.
    Menerima PATH NIfTI (bukan array) lalu panggil load_and_sync_masks
    persis seperti notebook.

    Returns
    -------
    grid_img     : np.ndarray (H, W, 3) uint8
    best_slice_z : int
    """
    dwi, pred_biner, _, _ = load_and_sync_masks(dwi_path, mask_path)

    n_sl    = pred_biner.shape[2]
    best_sl = int(np.argmax([pred_biner[:, :, z].sum() for z in range(n_sl)]))

    grid_img = plot_case(
        dwi=dwi,
        pred_biner=pred_biner,
        vol_ml=vol_ml,
        subject_id=subject_id,
        n_slices=n_display,
    )
    return grid_img, best_sl


def create_single_overlay(
    dwi_path: str,
    mask_path: str,
    slice_idx: Optional[int] = None,
) -> np.ndarray:
    """Best-slice side-by-side untuk thumbnail dashboard."""
    dwi, pred_biner, _, _ = load_and_sync_masks(dwi_path, mask_path)
    n_sl = pred_biner.shape[2]
    sl   = (int(np.argmax([pred_biner[:,:,z].sum() for z in range(n_sl)]))
            if slice_idx is None else max(0, min(n_sl-1, slice_idx)))

    dwi_sl  = dwi[:, :, sl].T
    pred_sl = pred_biner[:, :, sl].T
    disp    = normalize_for_display(dwi_sl)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.patch.set_facecolor('#0D1117')
    for ax in axes:
        ax.set_facecolor('#0D1117')
        ax.axis('off')
    axes[0].imshow(disp, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'DWI  sl.{sl}', color='#8B949E', fontsize=9)
    axes[1].imshow(red_overlay(disp, pred_sl, 0.65))
    axes[1].set_title('Lesion Overlay', color='#FF7B7B', fontsize=9)

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0D1117')
    plt.close(fig)
    buf.seek(0)
    result = np.array(Image.open(buf).convert('RGB'))
    buf.close()
    return result
