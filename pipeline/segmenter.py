"""pipeline/segmenter.py — wrapper NVAutoNet"""
import os
import torch
import SimpleITK as sitk
from models.nvautonet import NVAutoNet
from utils.nifti_utils import sitk_to_numpy_axial
from config import NVAUTONET_WEIGHTS

_segmenter_instance = None

def load_segmenter(device: str = None) -> NVAutoNet:
    global _segmenter_instance
    if _segmenter_instance is not None:
        return _segmenter_instance
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _segmenter_instance = NVAutoNet(weights_dir=str(NVAUTONET_WEIGHTS), device=device)
    return _segmenter_instance

def run_segmentation(adc_path: str, dwi_path: str, out_mask_path: str, device: str = None) -> dict:
    """
    Jalankan NVAutoNet. Selalu return dict dengan key lengkap.
    """
    _empty = {
        "mask_path": None, "mask_sitk": None,
        "mask_array": None, "prob_map": None,
        "success": False, "error": None,
    }

    try:
        seg = load_segmenter(device)
        pred_sitk, lesion_prob = seg.predict_and_save(
            adc_path=adc_path, dwi_path=dwi_path, out_path=out_mask_path,
        )
        mask_array = sitk_to_numpy_axial(pred_sitk)
        return {
            "mask_path": out_mask_path, "mask_sitk": pred_sitk,
            "mask_array": mask_array, "prob_map": lesion_prob,
            "success": True, "error": None,
        }

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            seg = load_segmenter(device)
            seg.checkpoints = seg.checkpoints[:5]
            pred_sitk, lesion_prob = seg.predict_and_save(
                adc_path=adc_path, dwi_path=dwi_path, out_path=out_mask_path,
            )
            mask_array = sitk_to_numpy_axial(pred_sitk)
            return {
                "mask_path": out_mask_path, "mask_sitk": pred_sitk,
                "mask_array": mask_array, "prob_map": lesion_prob,
                "success": True, "error": "GPU OOM: pakai 5 dari 15 checkpoint",
            }
        except Exception as e2:
            err = {**_empty}; err["error"] = f"OOM Fallback Error: {e2}"
            return err

    except Exception as e:
        err = {**_empty}; err["error"] = str(e)
        return err
EOF
echo "segmenter.py OK"
Output
