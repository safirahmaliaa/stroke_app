"""
pipeline/overlay.py
===================
Membuat visualisasi overlay lesion mask di atas citra DWI.
Dikonversi LANGSUNG dari notebook visualisasi_overlay_lengkap.ipynb
(fungsi normalize_for_display, red_overlay, plot_case).

Perbedaan kritis dari versi sebelumnya:
  - Slice diambil dengan transpose .T (notebook Cell 4)
  - Overlay RGBA manual, bukan matplotlib colormap
  - Normalisasi exclude background 0
  - Best slice = argmax jumlah voxel lesi (bukan brightness proxy)
  - Mask diresample ke DWI space via SimpleITK sebelum visualisasi
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Tuple, Optional


# ── Fungsi Inti (copy langsung dari notebook Cell 4) ──────────────────────────

def normalize_for_display(arr_2d: np.ndarray) -> np.ndarray:
    """
    Normalisasi slice 2D untuk display.
    Exclude background (nilai 0) saat menghitung percentile.
    Sesuai notebook: normalize_for_display()
    """
    brain = arr_2d > 0
    if brain.sum() == 0:
        return np.zeros_like(arr_2d, dtype=np.float32)
    p2, p98 = np.percentile(arr_2d[brain], [2, 98])
    out = np.clip((arr_2d - p2) / (p98 - p2 + 1e-8), 0, 1)
    out[~brain] = 0
    return out.astype(np.float32)


def red_overlay(disp: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Buat RGBA image dengan overlay merah di area lesi.
    Sesuai notebook: red_overlay()

    Parameters
    ----------
    disp    : np.ndarray (H, W) float [0,1], DWI grayscale
    mask_2d : np.ndarray (H, W) biner
    alpha   : transparansi overlay merah

    Returns
    -------
    np.ndarray (H, W, 4) float [0,1] RGBA
    """
    h, w = disp.shape
    rgb  = np.stack([disp, disp, disp], axis=-1)
    rgba = np.concatenate([rgb, np.ones((h, w, 1))], axis=-1).astype(np.float32)

    if mask_2d.sum() > 0:
        ov = np.zeros((h, w, 4), dtype=np.float32)
        ov[mask_2d == 1] = [1.0, 0.0, 0.0, alpha]
        a = ov[..., 3:4]
        rgba[..., :3] = rgba[..., :3] * (1 - a) + ov[..., :3] * a

    return np.clip(rgba, 0, 1)


def green_overlay(disp: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.60) -> np.ndarray:
    """Overlay hijau untuk ground truth. Sesuai notebook: green_overlay()"""
    h, w = disp.shape
    rgb  = np.stack([disp, disp, disp], axis=-1)
    rgba = np.concatenate([rgb, np.ones((h, w, 1))], axis=-1).astype(np.float32)

    if mask_2d.sum() > 0:
        ov = np.zeros((h, w, 4), dtype=np.float32)
        ov[mask_2d == 1] = [0.0, 1.0, 0.0, alpha]
        a = ov[..., 3:4]
        rgba[..., :3] = rgba[..., :3] * (1 - a) + ov[..., :3] * a

    return np.clip(rgba, 0, 1)


# ── Fungsi Utama ──────────────────────────────────────────────────────────────

def find_best_slice(pred_biner: np.ndarray) -> int:
    """
    Temukan slice dengan jumlah voxel lesi terbanyak.
    Sesuai notebook: argmax dari pred_biner[:,:,z].sum()
    """
    n_sl = pred_biner.shape[2]
    return int(np.argmax([pred_biner[:, :, z].sum() for z in range(n_sl)]))


def get_display_slices(pred_biner: np.ndarray, n_slices: int = 5) -> List[int]:
    """
    Pilih slice untuk ditampilkan: sekitar best slice dengan offset merata.
    Sesuai notebook: offsets = np.linspace(-8, 8, n_slices)
    """
    n_sl    = pred_biner.shape[2]
    best_sl = find_best_slice(pred_biner)
    offsets = np.linspace(-8, 8, n_slices, dtype=int)
    slices  = sorted(set([max(0, min(n_sl - 1, best_sl + o)) for o in offsets]))
    return slices


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
    Plot grid overlay merah sesuai notebook plot_case().

    Layout:
      Row 0 : DWI grayscale
      Row 1 : GT hijau (jika gt_biner tidak None)
      Row 2/1: Pred overlay merah

    KRITIS: slice diambil dengan .T sesuai notebook.

    Parameters
    ----------
    dwi        : (H, W, N) axial volume
    pred_biner : (H, W, N) biner mask
    vol_ml     : float volume lesi mL
    subject_id : label judul
    slices     : list indeks slice manual; None = otomatis
    n_slices   : jumlah slice jika otomatis
    gt_biner   : (H, W, N) ground truth biner, opsional

    Returns
    -------
    np.ndarray (H_out, W_out, 3) uint8 gambar hasil render
    """
    n_sl = pred_biner.shape[2]

    if slices is None:
        slices = get_display_slices(pred_biner, n_slices)

    best_sl = find_best_slice(pred_biner)
    n_rows  = 3 if gt_biner is not None else 2

    fig, axes = plt.subplots(
        n_rows, len(slices),
        figsize=(3.5 * len(slices), 3.2 * n_rows + 1.2)
    )
    fig.patch.set_facecolor('#0D1117')

    # Handle satu slice
    if len(slices) == 1:
        axes = axes.reshape(n_rows, 1)

    for col, sl in enumerate(slices):
        sl = max(0, min(n_sl - 1, sl))

        # KRITIS: pakai .T sesuai notebook
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
            # Row 1: GT hijau
            gt_sl = gt_biner[:, :, sl].T
            axes[1, col].imshow(green_overlay(disp, gt_sl, 0.60))
            if col == 0:
                axes[1, col].set_ylabel(
                    'Ground Truth', color='#7EC8A0', fontsize=10, labelpad=8
                )
            # Row 2: Pred merah
            axes[2, col].imshow(red_overlay(disp, pred_sl, 0.65))
            if col == 0:
                axes[2, col].set_ylabel(
                    'NVAUTO Pred.', color='#FF7B7B', fontsize=10, labelpad=8
                )
        else:
            # Row 1: Pred merah (tanpa GT)
            axes[1, col].imshow(red_overlay(disp, pred_sl, 0.65))
            if col == 0:
                axes[1, col].set_ylabel(
                    'Lesion Overlay', color='#FF7B7B', fontsize=10, labelpad=8
                )

    # Judul dengan severity
    sev   = 'RINGAN' if vol_ml < 10 else ('SEDANG' if vol_ml < 70 else 'BERAT')
    color = '#238636' if vol_ml < 10 else ('#d29922' if vol_ml < 70 else '#da3633')

    plt.suptitle(
        f'{subject_id}\nVol: {vol_ml:.2f} mL  |  {sev}',
        color='white', fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close(fig)
    buf.seek(0)
    result = np.array(Image.open(buf).convert('RGB'))
    buf.close()
    return result


def create_overlay_gallery(
    dwi_vol: np.ndarray,
    mask_vol: np.ndarray,
    vol_ml: float = 0.0,
    subject_id: str = "",
    n_display: int = 5,
) -> Tuple[List[np.ndarray], int]:
    """
    Buat galeri frame per-slice untuk ditampilkan di Streamlit.
    Setiap frame = satu kolom dari plot_case (DWI + overlay merah).

    Returns
    -------
    frames       : list of np.ndarray (H, W, 3) uint8
    best_slice_z : int indeks slice terbaik
    """
    n      = min(dwi_vol.shape[2], mask_vol.shape[2])
    slices = get_display_slices(mask_vol, n_display)
    best_z = find_best_slice(mask_vol)

    frames = []
    for sl in slices:
        sl = max(0, min(n - 1, sl))

        # KRITIS: .T sesuai notebook
        dwi_sl  = dwi_vol[:, :, sl].T
        pred_sl = mask_vol[:, :, sl].T
        disp    = normalize_for_display(dwi_sl)

        # Render satu frame (DWI + overlay)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        fig.patch.set_facecolor('#0D1117')

        for ax in axes:
            ax.set_facecolor('#0D1117')
            ax.axis('off')

        axes[0].imshow(disp, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'DWI sl.{sl}', color='#8B949E', fontsize=8)

        axes[1].imshow(red_overlay(disp, pred_sl, 0.65))
        axes[1].set_title('Lesion Overlay', color='#FF7B7B', fontsize=8)

        plt.tight_layout(pad=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor='#0D1117')
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf).convert('RGB')))
        buf.close()

    return frames, best_z


def create_full_grid(
    dwi_vol: np.ndarray,
    mask_vol: np.ndarray,
    vol_ml: float,
    subject_id: str = "MRI",
    n_slices: int = 5,
) -> np.ndarray:
    """
    Buat satu gambar grid lengkap (DWI + overlay merah per kolom).
    Cocok untuk ditampilkan sebagai satu gambar besar di hasil dashboard.

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    return plot_case(
        dwi=dwi_vol,
        pred_biner=mask_vol,
        vol_ml=vol_ml,
        subject_id=subject_id,
        slices=None,
        n_slices=n_slices,
    )
PYEOF
echo "overlay.py updated"
Output
