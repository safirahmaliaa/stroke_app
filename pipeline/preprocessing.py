"""
pipeline/preprocessing.py
==========================
Fungsi preprocessing slice MRI.
Dikonversi LANGSUNG dari notebook Cell 4 (preprocess_slice)
dan Cell 5 (logika Important Slices M3).

Pipeline (sesuai proposal Bab III):
  Step 1 : Resize 224×224
  Step 2 : Normalisasi [0, 1]
  Step 3a: CLAHE (clip_limit=0.02)
  Step 3b: Gaussian Filter (sigma=0.5)
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
from skimage.transform import resize
import torch
from torchvision import transforms

from config import (
    IMG_SIZE,
    CLAHE_CLIP_LIMIT,
    GAUSSIAN_SIGMA,
    BRAIN_COV_THRESH,
    OUTLIER_LOW_PCT,
    OUTLIER_HIGH_PCT,
    MAX_LESI_SLICES,
    MAX_NOLESI_SLICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils.nifti_utils import transpose_to_axial, resample_vol, resize_to


# Transform akhir: ImageNet normalize (sesuai notebook Cell 8, EVAL_TF)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_slice(sl: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocessing satu slice 2D sesuai notebook Cell 4.

    Pipeline:
    1. Pre-clip outlier (percentile 1-99 dari nonzero pixels)
    2. Resize ke size×size
    3. Normalisasi ke [0, 1]
    4. CLAHE (clip_limit=0.02)
    5. Gaussian filter (sigma=0.5)
    6. Re-normalize ke [0, 1]

    Parameters
    ----------
    sl   : numpy 2D float32
    size : target H = W = size

    Returns
    -------
    np.ndarray (size, size) float32, range [0, 1]
    """
    img = sl.copy().astype(np.float32)

    # Pre-clip outlier
    nz = img[img > 0]
    if len(nz) > 10:
        p_lo  = np.percentile(nz, OUTLIER_LOW_PCT)
        p_hi  = np.percentile(nz, OUTLIER_HIGH_PCT)
        img   = np.clip(img, p_lo, p_hi)

    # Step 1: Resize
    img = resize(
        img, (size, size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    # Step 2: Normalisasi [0, 1]
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    # Step 3a: CLAHE
    if img.max() > img.min():
        img = exposure.equalize_adapthist(
            img, clip_limit=CLAHE_CLIP_LIMIT
        ).astype(np.float32)

    # Step 3b: Gaussian filter
    img = ndi.gaussian_filter(img, sigma=GAUSSIAN_SIGMA).astype(np.float32)

    # Re-normalize setelah gaussian
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    return img


def slice_to_tensor(proc_slice: np.ndarray) -> torch.Tensor:
    """
    Konversi slice yang sudah dipreprocess → tensor siap inference.

    Pipeline:
    1. Replikasi 1 channel → 3 channel identik (DWI-only M3)
    2. ImageNet normalize

    Parameters
    ----------
    proc_slice : np.ndarray (H, W) float32

    Returns
    -------
    torch.Tensor (1, 3, H, W) float32
    """
    # Replikasi ke 3 channel (sesuai M3: DWI-only, 3ch identik)
    three_ch = np.stack([proc_slice, proc_slice, proc_slice], axis=0)  # (3, H, W)
    tensor   = torch.from_numpy(three_ch)
    tensor   = EVAL_TRANSFORM(tensor)       # ImageNet normalize
    return tensor.unsqueeze(0)             # (1, 3, H, W)


def extract_important_slices_dwi(
    dwi_vol: np.ndarray,
    n_lesi: int = MAX_LESI_SLICES,
    n_nolesi: int = MAX_NOLESI_SLICES,
) -> list:
    """
    Ekstrak "important slices" dari volume DWI untuk model M3.
    Karena saat inference tidak ada ground truth mask, kita gunakan
    proxy: slice dengan mean intensitas nonzero tertinggi dianggap
    paling informatif (berkorelasi dengan area lesi hyperintense di DWI).

    Parameters
    ----------
    dwi_vol  : np.ndarray (H, W, N) setelah transpose_to_axial
    n_lesi   : jumlah slice "suspicious" (proxy lesi) yang diambil
    n_nolesi : jumlah slice normal dari tengah volume

    Returns
    -------
    list of dict:
        {
          "z"       : int (indeks slice),
          "proc"    : np.ndarray (H, W) hasil preprocess,
          "tensor"  : torch.Tensor (1, 3, H, W),
          "score"   : float (brightness proxy),
          "is_susp" : bool (True = slice suspicious/informatif),
        }
    """
    n = dwi_vol.shape[2]
    all_slices = []

    for z in range(n):
        sl = dwi_vol[:, :, z]

        # Filter slice hampir kosong (brain coverage < 5%)
        brain_cov = np.count_nonzero(sl) / sl.size
        if brain_cov < BRAIN_COV_THRESH:
            continue

        # Brightness proxy: mean intensitas nonzero
        nz     = sl[sl > 0]
        score  = float(np.mean(nz)) if len(nz) > 0 else 0.0
        proc   = preprocess_slice(sl)
        tensor = slice_to_tensor(proc)

        all_slices.append({
            "z"     : z,
            "proc"  : proc,
            "tensor": tensor,
            "score" : score,
        })

    if not all_slices:
        return []

    # Urutkan berdasarkan brightness score (descending)
    sorted_slices = sorted(all_slices, key=lambda x: x["score"], reverse=True)

    # Top-n sebagai "suspicious" (proxy untuk slice ada lesi)
    top_n       = sorted_slices[:n_lesi]
    top_n_idx   = {s["z"] for s in top_n}

    # Middle slices sebagai "normal"
    remaining   = [s for s in all_slices if s["z"] not in top_n_idx]
    mid         = len(remaining) // 2
    half        = n_nolesi // 2
    mid_slices  = remaining[max(0, mid - half): min(len(remaining), mid + half)]
    if not mid_slices:
        mid_slices = remaining[:n_nolesi]

    # Gabungkan dan tandai
    result = []
    for s in top_n:
        result.append({**s, "is_susp": True})
    for s in mid_slices:
        result.append({**s, "is_susp": False})

    return result


def prepare_clinical_volumes(
    dwi_vol: np.ndarray,
    adc_vol: np.ndarray = None,
    flair_vol: np.ndarray = None,
) -> tuple:
    """
    Siapkan volume-volume MRI untuk inference.

    1. Transpose semua ke axial (H, W, N)
    2. Resample ADC dan FLAIR ke jumlah slice DWI
    3. Extract important slices dari DWI

    Parameters
    ----------
    dwi_vol   : np.ndarray (X, Y, Z) DWI raw
    adc_vol   : np.ndarray (X, Y, Z) ADC raw, opsional
    flair_vol : np.ndarray (X, Y, Z) FLAIR raw, opsional

    Returns
    -------
    dwi_axial  : np.ndarray (H, W, N)
    adc_axial  : np.ndarray (H, W, N) atau None
    flair_axial: np.ndarray (H, W, N) atau None
    slices     : list of dict dari extract_important_slices_dwi
    """
    dwi_axial   = transpose_to_axial(dwi_vol)
    n           = dwi_axial.shape[2]

    adc_axial   = None
    flair_axial = None

    if adc_vol is not None:
        adc_axial = transpose_to_axial(adc_vol)
        if adc_axial.shape[2] != n:
            adc_axial = resample_vol(adc_axial, n)

    if flair_vol is not None:
        flair_axial = transpose_to_axial(flair_vol)
        if flair_axial.shape[2] != n:
            flair_axial = resample_vol(flair_axial, n)

    slices = extract_important_slices_dwi(dwi_axial)

    return dwi_axial, adc_axial, flair_axial, slices