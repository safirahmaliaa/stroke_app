"""
pipeline/overlay.py
===================

Visualisasi overlay lesion sesuai notebook nvauto_fix_v2.

Output:
    frames : list[np.ndarray(H,W,3)]
"""

import numpy as np
from typing import List, Tuple


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalisasi citra DWI seperti notebook.
    """

    img = img.astype(np.float32)

    nz = img[img > 0]

    if len(nz) == 0:
        return np.zeros_like(img, dtype=np.float32)

    p2 = np.percentile(nz, 2)
    p98 = np.percentile(nz, 98)

    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

    return img.astype(np.float32)


def red_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Overlay merah identik notebook.
    """

    rgb = np.stack([img, img, img], axis=-1)

    lesion = mask > 0

    rgb[..., 0] = np.where(
        lesion,
        rgb[..., 0] * (1 - alpha) + alpha,
        rgb[..., 0],
    )

    rgb[..., 1] = np.where(
        lesion,
        rgb[..., 1] * (1 - alpha),
        rgb[..., 1],
    )

    rgb[..., 2] = np.where(
        lesion,
        rgb[..., 2] * (1 - alpha),
        rgb[..., 2],
    )

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb


def find_best_slices(
    mask_vol: np.ndarray,
    n_display: int = 5,
) -> List[int]:
    """
    Ambil slice dengan area lesi terbesar.
    """

    n_slice = mask_vol.shape[2]

    lesion_area = [
        int(mask_vol[:, :, z].sum())
        for z in range(n_slice)
    ]

    if max(lesion_area) == 0:

        mid = n_slice // 2

        start = max(0, mid - n_display // 2)

        end = min(n_slice, start + n_display)

        return list(range(start, end))

    best = int(np.argmax(lesion_area))

    offsets = [-6, -3, 0, 3, 6]

    idx = []

    for off in offsets:

        z = best + off

        z = max(0, min(n_slice - 1, z))

        if z not in idx:
            idx.append(z)

    return idx[:n_display]


def create_overlay_gallery(
    dwi_vol: np.ndarray,
    mask_vol: np.ndarray,
    n_display: int = 5,
) -> Tuple[List[np.ndarray], int]:

    slice_idx = find_best_slices(
        mask_vol,
        n_display=n_display,
    )

    frames = []

    for z in slice_idx:

        dwi = normalize_for_display(
            dwi_vol[:, :, z].T
        )

        mask = mask_vol[:, :, z].T

        overlay = red_overlay(
            dwi,
            mask,
            alpha=0.65,
        )

        frames.append(overlay)

    if len(slice_idx):

        best_slice = slice_idx[len(slice_idx)//2]

    else:

        best_slice = mask_vol.shape[2] // 2

    return frames, best_slice

def create_side_by_side(
    dwi_slice: np.ndarray,
    mask_slice: np.ndarray,
    title: str = "",
) -> np.ndarray:
    """
    Membuat gambar perbandingan:
    kiri  = DWI
    kanan = DWI + Overlay
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    dwi = normalize_for_display(dwi_slice.T)
    mask = mask_slice.T

    overlay = red_overlay(dwi, mask)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(dwi, cmap="gray")
    ax[0].set_title("DWI")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title("Overlay")
    ax[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)