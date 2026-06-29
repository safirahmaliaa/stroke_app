"""
pipeline/segmenter.py
=====================
Wrapper pipeline untuk menjalankan segmentasi NVAutoNet
dari file NIfTI yang diupload user.
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
import traceback
from models.nvautonet import NVAutoNet
from utils.nifti_utils import sitk_to_numpy_axial
from config import NVAUTONET_WEIGHTS


# Singleton segmenter
_segmenter_instance = None


def load_segmenter(device: str = None) -> NVAutoNet:
    """
    Load NVAutoNet dengan singleton pattern.
    """
    global _segmenter_instance

    if _segmenter_instance is not None:
        return _segmenter_instance

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _segmenter_instance = NVAutoNet(
        weights_dir=str(NVAUTONET_WEIGHTS),
        device=device,
    )
    return _segmenter_instance


def run_segmentation(
    adc_path: str,
    dwi_path: str,
    out_mask_path: str,
    device: str = None,
) -> dict:
    """
    Jalankan segmentasi NVAutoNet dan kembalikan hasil lengkap.

    Parameters
    ----------
    adc_path      : path ADC NIfTI
    dwi_path      : path DWI NIfTI
    out_mask_path : path simpan mask hasil (.nii.gz)
    device        : "cuda" | "cpu" | None (auto)

    Returns
    -------
    dict:
        "mask_path"   : str, path ke file mask tersimpan
        "mask_sitk"   : sitk.Image
        "mask_array"  : np.ndarray (H, W, N) biner uint8, axial
        "prob_map"    : np.ndarray probabilitas lesi
        "success"     : bool
        "error"       : str | None
    """
    try:
        segmenter = load_segmenter(device)
        pred_sitk, lesion_prob = segmenter.predict_and_save(
            adc_path=adc_path,
            dwi_path=dwi_path,
            out_path=out_mask_path,
        )

        # Konversi ke numpy axial untuk visualisasi
        mask_array = sitk_to_numpy_axial(pred_sitk)

        return {
            "mask_path" : out_mask_path,
            "mask_sitk" : pred_sitk,
            "mask_array": mask_array,
            "prob_map"  : lesion_prob,
            "success"   : True,
            "error"     : None,
        }

    except torch.cuda.OutOfMemoryError:
        # Fallback: kurangi jumlah checkpoint jika OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            seg = load_segmenter(device)
            seg.checkpoints = seg.checkpoints[:5]   # pakai 5 checkpoint saja
            pred_sitk, lesion_prob = seg.predict_and_save(
                adc_path=adc_path,
                dwi_path=dwi_path,
                out_path=out_mask_path,
            )
            mask_array = sitk_to_numpy_axial(pred_sitk)
            return {
                "mask_path" : out_mask_path,
                "mask_sitk" : pred_sitk,
                "mask_array": mask_array,
                "prob_map"  : lesion_prob,
                "success"   : True,
                "error"     : "GPU OOM: digunakan 5 checkpoint (dari 15)",
            }
        except Exception as e2:
            return {"success": False, "error": f"OOM Fallback Error: {e2}"}
    
    except Exception as e:
        return {
            "success": False,
            "error"  : str(e),
            "mask_path"  : None,
            "mask_sitk"  : None,
            "mask_array" : None,
            "prob_map"   : None,
        }