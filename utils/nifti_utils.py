"""
utils/nifti_utils.py
====================
Fungsi bantu untuk baca/tulis file NIfTI.
Diadaptasi dari notebook Cell 3 (fungsi load_nifti, find_isles_files,
resample_vol, resize_to).
"""

import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from typing import Optional, Dict, Tuple


def load_nifti(path: str) -> Optional[np.ndarray]:
    """
    Load file NIfTI → numpy float32 dengan orientasi canonical.
    Jika volume 4D, ambil volume terakhir (indeks -1).

    Parameters
    ----------
    path : str
        Path ke file .nii atau .nii.gz.

    Returns
    -------
    np.ndarray atau None jika gagal load.
    """
    try:
        img  = nib.as_closest_canonical(nib.load(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[:, :, :, -1]
        return data
    except Exception:
        return None


def get_nifti_spacing(path: str) -> Tuple[float, float, float]:
    """
    Ambil voxel spacing (mm) dari header NIfTI menggunakan SimpleITK.

    Returns
    -------
    (sx, sy, sz) dalam satuan mm.
    """
    img = sitk.ReadImage(path)
    if img.GetDimension() == 4:
        # Untuk 4D, ambil 3 komponen pertama spacing
        return tuple(img.GetSpacing()[:3])
    return img.GetSpacing()


def resample_vol(src: np.ndarray, n_target: int) -> np.ndarray:
    """
    Resample jumlah slice pada sumbu Z ke n_target menggunakan
    interpolasi linear. Digunakan untuk menyamakan jumlah slice
    ADC/FLAIR dengan DWI.

    Parameters
    ----------
    src      : volume (H, W, N_original)
    n_target : jumlah slice yang diinginkan

    Returns
    -------
    np.ndarray (H, W, n_target)
    """
    if src.shape[2] == n_target:
        return src

    idx = np.linspace(0, src.shape[2] - 1, n_target)
    out = np.zeros((src.shape[0], src.shape[1], n_target), dtype=np.float32)
    for i, x in enumerate(idx):
        lo = int(x)
        hi = min(lo + 1, src.shape[2] - 1)
        out[:, :, i] = (1 - (x - lo)) * src[:, :, lo] + (x - lo) * src[:, :, hi]
    return out


def resize_to(sl: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize slice 2D ke target_shape jika berbeda menggunakan
    anti-aliased resize dari skimage.

    Parameters
    ----------
    sl           : slice 2D numpy
    target_shape : tuple (H, W)
    """
    if sl.shape == target_shape:
        return sl
    return resize(
        sl.astype(np.float32),
        target_shape,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def transpose_to_axial(vol: np.ndarray) -> np.ndarray:
    """
    Transpose volume dari (X, Y, Z) → (H, W, N) untuk orientasi axial.
    Ini sesuai dengan konvensi notebook (Cell 5, baris `def T(v)`).
    """
    return np.transpose(vol, (1, 0, 2))


def find_best_axial_slice(vol: np.ndarray, mode: str = "brightness") -> int:
    """
    Temukan indeks slice axial terbaik dari volume (H, W, N).

    Parameters
    ----------
    vol  : numpy array (H, W, N)
    mode : "brightness" → slice dengan mean intensitas tertinggi
           "center"     → slice tengah volume

    Returns
    -------
    int : indeks slice terbaik
    """
    n = vol.shape[2]
    if mode == "center":
        return n // 2
    # mode brightness: slice dengan rata-rata intensitas nonzero tertinggi
    scores = []
    for z in range(n):
        sl = vol[:, :, z]
        nz = sl[sl > 0]
        scores.append(float(np.mean(nz)) if len(nz) > 0 else 0.0)
    return int(np.argmax(scores))


def resample_mask_to_reference(
    mask_path: str,
    ref_path: str,
) -> np.ndarray:
    """
    Resample lesion mask ke ruang koordinat fisik referensi (DWI).
    Ini adalah fungsi kunci untuk evaluasi Dice yang benar,
    sesuai notebook nvauto_fix_v2.ipynb Cell 4 (dice_physical_space).

    Parameters
    ----------
    mask_path : path ke file mask NIfTI
    ref_path  : path ke file referensi (biasanya DWI)

    Returns
    -------
    np.ndarray biner (H, W, D) dalam ruang referensi
    """
    mask_sitk = sitk.ReadImage(mask_path)
    ref_sitk  = sitk.ReadImage(ref_path)

    if ref_sitk.GetDimension() == 4:
        size_3d = list(ref_sitk.GetSize()[:3]) + [0]
        ref_sitk = sitk.Extract(ref_sitk, size_3d, [0, 0, 0, 0])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(mask_sitk)

    arr = sitk.GetArrayFromImage(resampled)
    return (arr > 0).astype(np.uint8)


def sitk_to_numpy_axial(sitk_img: sitk.Image) -> np.ndarray:
    """
    Konversi SimpleITK Image → numpy array dalam orientasi axial (H, W, N).
    SimpleITK menggunakan (Z, Y, X) ordering, kita transpose ke (X, Y, Z)
    lalu ke axial (Y, X, Z).
    """
    arr = sitk.GetArrayFromImage(sitk_img)   # (Z, Y, X)
    return arr.transpose(2, 1, 0)            # (X, Y, Z) → (H, W, N) axial