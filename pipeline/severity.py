"""
pipeline/severity.py
====================
Estimasi tingkat keparahan stroke berdasarkan volume lesi.
Threshold keparahan diambil dari notebook nvauto_fix_v2.ipynb Cell 10
(visualisasi menggunakan label RINGAN/SEDANG/BERAT).

Referensi klinis:
  - Olivot et al. (2009) mendeskripsikan korelasi volume lesi DWI
    dengan outcome klinis pasien stroke iskemik.
  - Threshold 10 mL dan 70 mL merupakan nilai yang umum digunakan
    dalam penelitian stroke untuk stratifikasi keparahan.
"""

from config import (
    SEVERITY_RINGAN_MAX,
    SEVERITY_SEDANG_MAX,
    SEVERITY_LABELS,
)


def estimate_severity(volume_ml: float) -> dict:
    """
    Tentukan tingkat keparahan stroke berdasarkan volume lesi.

    Kriteria (dari notebook):
      RINGAN : volume < 10 mL
      SEDANG : 10 mL ≤ volume ≤ 70 mL
      BERAT  : volume > 70 mL

    Parameters
    ----------
    volume_ml : float, volume lesi dalam mL

    Returns
    -------
    dict:
        "key"        : "RINGAN" | "SEDANG" | "BERAT"
        "label"      : str label tampil
        "color"      : str hex color untuk UI
        "description": str deskripsi klinis
        "emoji"      : str emoji indikator
        "volume_ml"  : float volume input
    """
    if volume_ml < SEVERITY_RINGAN_MAX:
        key = "RINGAN"
    elif volume_ml <= SEVERITY_SEDANG_MAX:
        key = "SEDANG"
    else:
        key = "BERAT"

    result = SEVERITY_LABELS[key].copy()
    result["key"]       = key
    result["volume_ml"] = volume_ml

    return result