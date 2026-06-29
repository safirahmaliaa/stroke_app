"""
pipeline/volume.py
==================
Kalkulasi volume lesi dari lesion mask NIfTI.
Formula sesuai notebook nvauto_fix_v2.ipynb Cell 4 (fungsi dice_physical_space):

    vol_ml = n_voxels_positif × sx × sy × sz / 1000

di mana sx, sy, sz adalah voxel spacing dalam satuan mm.
"""

import numpy as np
import SimpleITK as sitk


def calculate_volume_ml(
    mask_sitk: sitk.Image,
) -> float:
    """
    Hitung volume lesi dalam mililiter (mL) dari SimpleITK Image.

    Volume = jumlah voxel positif × volume satu voxel (mm³) / 1000

    Parameters
    ----------
    mask_sitk : sitk.Image
        Lesion mask biner hasil NVAutoNet.

    Returns
    -------
    float : volume lesi dalam mL, dibulatkan 2 desimal.
    """
    spacing  = mask_sitk.GetSpacing()   # (sx, sy, sz) dalam mm
    arr      = sitk.GetArrayFromImage(mask_sitk)
    n_voxels = int((arr > 0).sum())

    # Volume satu voxel dalam mm³
    voxel_vol_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])

    # Konversi ke mL (1 mL = 1000 mm³)
    volume_ml = (n_voxels * voxel_vol_mm3) / 1000.0

    return round(volume_ml, 2)


def calculate_volume_from_array(
    mask_array: np.ndarray,
    spacing_mm: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    Hitung volume lesi dari numpy array mask.
    Digunakan sebagai fallback jika SimpleITK Image tidak tersedia.

    Parameters
    ----------
    mask_array : np.ndarray biner, shape bebas
    spacing_mm : tuple (sx, sy, sz) dalam mm

    Returns
    -------
    float : volume dalam mL
    """
    n_voxels    = int((mask_array > 0).sum())
    voxel_vol   = float(spacing_mm[0]) * float(spacing_mm[1]) * float(spacing_mm[2])
    return round((n_voxels * voxel_vol) / 1000.0, 2)