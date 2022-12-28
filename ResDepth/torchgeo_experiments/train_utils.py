import os
from typing import Any, Dict
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO hardcoded from 2021 experiments, should check if these are still appropriate
# dsm_mean_std = {"mean": None, "std": 33.336839170631755}
# dsm_std = dsm_mean_std["std"]
# ortho_mean_std = {"mean": 261.1999816894531, "std": 245.63670349121094}
# ortho_mean = ortho_mean_std["mean"]
# ortho_std = ortho_mean_std["std"]

# Stats for the subset of tiles that excludes forest & other areas with very large errors
# small_errors_only_mosaic_TRAIN_1020010042D39D00.r100_ortho_1.0m_ba.tif: mean=293.65, std=425.82
# small_errors_only_mosaic_TRAIN_1020010043455300.r100_ortho_1.0m_ba.tif: mean=262.90, std=409.16
# small_errors_only_mosaic_TRAIN_WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-IntersectionErr.tif: mean=0.12, std=0.15
# small_errors_only_mosaic_TRAIN_WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM.tif: mean=1031.83, std=1016.49
# small_errors_only_mosaic_TRAIN_USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif: mean=1052.88, std=1017.68
dsm_std = 1016
ortho_mean = 293
ortho_std = 417
trierror_mean = 0.12
trierror_std = 0.15


def plot_batch(inputs, filename=None):
    """Visualize inputs to the NN just before ingestion"""
    inputs = inputs.cpu().numpy()
    batch_size = inputs.shape[0]
    ncols = 4
    fig, ax = plt.subplots(batch_size, ncols, figsize=(ncols * 4, 2 * batch_size))
    plt.suptitle(
        f"Input tiles: Batch of size {inputs.shape[0]}"
    )  # \nCheckpoint: {ckpt}\nDataset: {val_directory}, Tiles=???")

    plt.rcParams["figure.dpi"] = 300  # maybe excessive

    for i in range(batch_size):
        initial_dem = inputs[i][0].squeeze()
        ax[i][0].imshow(hillshade(initial_dem), cmap="gray", rasterized=True)
        im = ax[i][0].imshow(initial_dem, alpha=0.5, rasterized=True)
        ax[i][0].set_title("initial DEM")
        plt.colorbar(im, ax=ax[i][0], fraction=0.04)

        im = ax[i][1].imshow(inputs[i][1].squeeze(), cmap="gray", rasterized=True)
        ax[i][1].set_title("ortho left")
        plt.colorbar(im, ax=ax[i][1], fraction=0.04)

        im = ax[i][2].imshow(inputs[i][2].squeeze(), cmap="gray", rasterized=True)
        ax[i][2].set_title("ortho right")
        plt.colorbar(im, ax=ax[i][2], fraction=0.04)

        im = ax[i][3].imshow(inputs[i][3].squeeze(), cmap="gray", rasterized=True)
        ax[i][3].set_title("nodata mask")
        plt.colorbar(im, ax=ax[i][3], fraction=0.04)

    plt.tight_layout()

    timestamp = datetime.datetime.isoformat(datetime.datetime.now())
    os.makedirs("batches", exist_ok=True)
    if filename:
        plt.savefig(filename)
    else:
        plt.savefig(f"batches/batch_{timestamp}.png")
    plt.close()


def hillshade(array, azimuth=315, angle_altitude=45):
    """Borrow hillshade

    https://github.com/uw-cryo/wv_stereo_processing/blob/54f3e82f313773e57dea8c3af5f05bb69e6b0a68/notebooks/gm_aso_dg_comparison.ipynb"""

    # Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html

    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
        slope
    ) * np.cos(azimuthrad - aspect)
    return 255 * (shaded + 1) / 2


# def normalize_orthos_and_dsm(sample: Dict[str, Any]) -> Dict[str, Any]:
#     """Apply the dsm and ortho normalization.

#     Args:
#         sample: dictionary from torchgeo

#     Returns
#         sample with normalized
#     """


def remove_bbox(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the bounding box property from a sample.

    Args:
        sample: dictionary with geographic metadata

    Returns
        sample without the bbox property
    """
    del sample["bbox"]
    return sample


def call_model(model, model_input_tensor: torch.Tensor):
    """Helper: given pytorch model & input rasters, return model output as numpy array"""
    # tile_output = model(model_input_tensor).detach().numpy().squeeze()
    tile_output = model(model_input_tensor).squeeze()
    return tile_output
