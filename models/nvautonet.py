"""
models/nvautonet.py
===================
Wrapper NVAUTOFixed untuk inferensi segmentasi lesi stroke.
Dikonversi LANGSUNG dari notebook nvauto_fix_v2.ipynb (Cell 7),
tanpa perubahan logika apapun.

Model ini menggunakan ensemble 15 TorchScript checkpoint (.ts)
dari framework DeepISLES untuk segmentasi volumetrik 3D.

Input  : path ADC.nii.gz + DWI.nii.gz
Output : SimpleITK Image (lesion mask biner) + probability map numpy
"""

import os
import glob
import tempfile

import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast

from monai import transforms, data
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch

from config import (
    NVAUTONET_WEIGHTS,
    NVAUTO_ROI_SIZE,
    NVAUTO_OVERLAP,
    NVAUTO_SW_BATCH,
    NVAUTO_SPACING,
    NVAUTO_SEG_THRESH,
)


class NVAutoNet:
    """
    Wrapper untuk model segmentasi NVAutoNet (DeepISLES framework).

    Cara kerja:
    1. Load semua TorchScript checkpoint (.ts) dari weights_dir.
    2. Untuk setiap kasus, jalankan inference dengan semua checkpoint.
    3. Rata-ratakan probabilitas (ensemble), lalu threshold → mask biner.
    4. Simpan mask dengan metadata spatial dari DWI asli.

    Catatan: Model ini memerlukan GPU. CPU inference sangat lambat
             karena SlidingWindowInferer dengan roi=[192,192,128].
    """

    def __init__(self, weights_dir: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights_dir is None:
            weights_dir = str(NVAUTONET_WEIGHTS)

        self.checkpoints = sorted(glob.glob(os.path.join(weights_dir, "model*.ts")))
        if not self.checkpoints:
            raise FileNotFoundError(
                f"Tidak ada file model*.ts di '{weights_dir}'.\n"
                f"Pastikan folder weights/nvautonet/ts/ berisi "
                f"model0.ts hingga model14.ts."
            )

        print(f"[NVAutoNet] Loaded {len(self.checkpoints)} checkpoint(s) dari {weights_dir}")

        # MONAI preprocessing pipeline (sesuai notebook Cell 7)
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.CastToTyped(keys=["image"], dtype=np.float32),
            transforms.EnsureTyped(keys=["image"], data_type="tensor"),
            transforms.Spacingd(
                keys=["image"],
                pixdim=NVAUTO_SPACING,
                mode=["bilinear"],
            ),
            transforms.NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
        ])

        # SlidingWindowInferer (sesuai notebook Cell 7)
        self.inferer = SlidingWindowInferer(
            roi_size=NVAUTO_ROI_SIZE,
            overlap=NVAUTO_OVERLAP,
            mode="gaussian",
            cache_roi_weight_map=True,
            sw_batch_size=NVAUTO_SW_BATCH,
        )

    def _get_3d_sitk(self, path: str) -> sitk.Image:
        """Load NIfTI sebagai 3D SimpleITK Image. Jika 4D, ambil volume terakhir."""
        img = sitk.ReadImage(path)
        if img.GetDimension() == 4:
            size_3d = list(img.GetSize()[:3]) + [0]
            img = sitk.Extract(img, size_3d, [0, 0, 0, 0])
        return img

    def _ensure_3d_file(self, path: str) -> tuple:
        """
        Pastikan file NIfTI adalah 3D.
        Jika 4D, tulis versi 3D ke file temp dan kembalikan pathnya.
        Return (path_to_use, was_converted).
        """
        img = sitk.ReadImage(path)
        if img.GetDimension() == 4:
            tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            tmp_path = tmp.name
            tmp.close()
            vol = img.GetSize()[3] - 1
            size_3d = list(img.GetSize()[:3]) + [0]
            sitk.WriteImage(
                sitk.Extract(img, size_3d, [0, 0, 0, vol]),
                tmp_path,
            )
            return tmp_path, True
        return path, False

    def predict(self, adc_path: str, dwi_path: str) -> tuple:
        """
        Jalankan inference NVAutoNet pada satu kasus.

        Parameters
        ----------
        adc_path : str
            Path ke file ADC NIfTI (.nii / .nii.gz).
        dwi_path : str
            Path ke file DWI NIfTI (.nii / .nii.gz).

        Returns
        -------
        pred_sitk : sitk.Image
            Lesion mask biner dengan metadata spatial dari DWI asli.
        lesion_prob : np.ndarray
            Probability map (H, W, D) di native space.
        """
        torch.backends.cudnn.benchmark = True

        # Pastikan DWI adalah 3D
        dwi_3d, was_converted = self._ensure_3d_file(dwi_path)

        try:
            # Buat MONAI dataset
            ds = data.Dataset(
                data=[{"image": [adc_path, dwi_3d]}],
                transform=self.transform,
            )
            dl = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            with torch.no_grad():
                for batch in dl:
                    image = batch["image"].to(self.device)
                    all_probs = []

                    for ckpt_path in self.checkpoints:

                        model = torch.jit.load(
                            ckpt_path,
                            map_location=torch.device(self.device),
                        )
                        model.eval()

                        if self.device == "cuda":
                            model = model.cuda()
                            image = image.cuda()

                            with autocast():
                                logits = self.inferer(
                                    inputs=image,
                                    network=model,
                                )
                        else:
                            model = model.cpu()
                            image = image.cpu()

                            logits = self.inferer(
                                inputs=image,
                                network=model,
                            )

                        probs = torch.softmax(logits.float(), dim=1)
                        batch["pred"] = probs

                        inverter = transforms.Invertd(
                            keys="pred",
                            transform=self.transform,
                            orig_keys="image",
                            meta_keys="pred_meta_dict",
                            nearest_interp=False,
                            to_tensor=True,
                        )

                        per_item = [
                            inverter(x)["pred"]
                            for x in decollate_batch(batch)
                        ]
                        all_probs.append(torch.stack(per_item).cpu())

                        del model

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Ensemble: rata-rata semua checkpoint
                    prob_mean   = sum(all_probs) / len(all_probs)
                    lesion_prob = prob_mean[0, 1].numpy()   # channel 1 = lesi
                    labels      = (lesion_prob > NVAUTO_SEG_THRESH).astype(np.int8)

            # Buat output SimpleITK Image dengan metadata dari DWI
            ref = self._get_3d_sitk(dwi_path)
            out = sitk.GetImageFromArray(labels.transpose(2, 1, 0))
            out.CopyInformation(ref)

            return out, lesion_prob

        finally:
            # Bersihkan file temp jika dibuat
            if was_converted and os.path.exists(dwi_3d):
                os.remove(dwi_3d)

    def predict_and_save(self, adc_path: str, dwi_path: str, out_path: str) -> tuple:
        """
        Predict dan simpan mask ke file NIfTI.

        Parameters
        ----------
        adc_path  : path ADC
        dwi_path  : path DWI
        out_path  : path output mask (.nii.gz)

        Returns
        -------
        pred_sitk : sitk.Image
        lesion_prob : np.ndarray
        """
        pred_sitk, lesion_prob = self.predict(adc_path, dwi_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sitk.WriteImage(pred_sitk, out_path)
        return pred_sitk, lesion_prob

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoints)

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"