"""
utils/image_utils.py
====================
Fungsi bantu untuk memproses gambar 2D (JPG/PNG)
pada Mode Quick Analysis.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


# Transform untuk inference (tanpa augmentasi)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_image_2d(uploaded_file) -> np.ndarray:
    """
    Load file gambar dari Streamlit UploadedFile → numpy RGB uint8.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile object

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img, dtype=np.uint8)


def prepare_image_tensor(img_array: np.ndarray) -> torch.Tensor:
    """
    Konversi numpy image (H, W, 3) uint8 → tensor siap inference.

    Pipeline:
    1. Resize ke 224×224
    2. Normalize ke [0, 1]
    3. Transpose ke (3, H, W)
    4. ImageNet normalize

    Parameters
    ----------
    img_array : np.ndarray (H, W, 3) uint8

    Returns
    -------
    torch.Tensor (1, 3, 224, 224) float32
    """
    # Resize
    img = Image.fromarray(img_array).resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]

    # (H, W, 3) → (3, H, W)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))

    # ImageNet normalize
    tensor = EVAL_TRANSFORM(tensor)

    # Tambah batch dimension
    return tensor.unsqueeze(0)   # (1, 3, 224, 224)


def resize_for_display(img_array: np.ndarray,
                        max_size: int = 512) -> np.ndarray:
    """
    Resize gambar untuk ditampilkan di Streamlit
    dengan mempertahankan aspect ratio.

    Parameters
    ----------
    img_array : np.ndarray (H, W, 3)
    max_size  : ukuran sisi terpanjang maksimal

    Returns
    -------
    np.ndarray (H', W', 3)
    """
    h, w = img_array.shape[:2]
    if max(h, w) <= max_size:
        return img_array

    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)