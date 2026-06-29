"""
pipeline/classifier.py
======================
Load model M3 (DenseNet121+CBAM) dan jalankan inference klasifikasi biner.
"""

import os
import torch
import numpy as np
from typing import List, Dict

from models.densenet_cbam import DenseNet121CBAM
from config import CLASSIFIER_WEIGHTS, CLASSIFY_THRESHOLD, DROPOUT

_model_instance = None


def load_classifier(device: str = None) -> DenseNet121CBAM:
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = str(CLASSIFIER_WEIGHTS)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan:\n{weights_path}"
        )

    model = DenseNet121CBAM(
        dropout=DROPOUT,
        freeze_backbone=False,
    )

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    # -------------------------------------------------------
    # Ambil state_dict
    # -------------------------------------------------------
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "state" in checkpoint:
            state_dict = checkpoint["state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # -------------------------------------------------------
    # Rename key checkpoint lama -> model sekarang
    # -------------------------------------------------------
    new_state_dict = {}

    for key, value in state_dict.items():

        key = key.replace("cbam.ca.", "cbam.channel_att.")
        key = key.replace("cbam.sa.", "cbam.spatial_att.")

        new_state_dict[key] = value

    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    _model_instance = model

    print(f"[Classifier] Loaded on {device}")

    return model


@torch.no_grad()
def predict_slices(slice_list: List[Dict], device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_classifier(device)

    results = []

    for s in slice_list:

        tensor = s["tensor"].to(device)

        logit = model(tensor)

        prob = torch.sigmoid(logit).item()

        results.append((s["z"], prob))

    if len(results) == 0:
        return {
            "label": "Non-Lesi",
            "probability": 0.0,
            "confidence": 1.0,
            "per_slice": [],
            "n_lesi": 0,
            "n_total": 0,
        }

    probs = [p for _, p in results]

    max_prob = float(max(probs))
    mean_prob = float(np.mean(probs))
    n_lesi = sum(p >= CLASSIFY_THRESHOLD for p in probs)

    label = "Lesi" if max_prob >= CLASSIFY_THRESHOLD else "Non-Lesi"

    confidence = abs(max_prob - CLASSIFY_THRESHOLD) / max(
        CLASSIFY_THRESHOLD,
        1 - CLASSIFY_THRESHOLD,
    )

    confidence = float(np.clip(confidence * 2, 0, 1))

    return {
        "label": label,
        "probability": max_prob,
        "mean_prob": mean_prob,
        "confidence": confidence,
        "per_slice": results,
        "n_lesi": n_lesi,
        "n_total": len(results),
    }


@torch.no_grad()
def predict_single_image(tensor, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_classifier(device)

    tensor = tensor.to(device)

    logit = model(tensor)

    prob = float(torch.sigmoid(logit).item())

    label = "Lesi" if prob >= CLASSIFY_THRESHOLD else "Non-Lesi"

    confidence = abs(prob - CLASSIFY_THRESHOLD) / max(
        CLASSIFY_THRESHOLD,
        1 - CLASSIFY_THRESHOLD,
    )

    confidence = float(np.clip(confidence * 2, 0, 1))

    return {
        "label": label,
        "probability": prob,
        "confidence": confidence,
    }