"""Evaluate trained model from checkpoint"""
import sys
import os

import numpy as np
import math

import torch
import torchvision

from torchgeo.samplers import GridGeoSampler  # appropriate for tile based dataset
from torchgeo.datasets import stack_samples

from torch.utils.data import DataLoader

import rasterio
from rasterio.transform import from_origin

# TODO figure out how to change to relative import of ResDepth UNet
sys.path.insert(0, "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/ResDepth/")
from lib.UNet import UNet

# Import our dataset
from torchgeo_dataset import TGDSMOrthoDataset

# Imports from train.py
from train_utils import (
    # remove_bbox,
    dsm_std,
    ortho_mean,
    ortho_std,
    trierror_mean,
    trierror_std,
    call_model,
)

TILE_SIZE = 256

if __name__ == "__main__":
    # Command-line args
    ckpt = sys.argv[1]
    normalization_method = sys.argv[2]
    if len(sys.argv) > 3:
        output_filename = sys.argv[3]
    else:
        output_filename = "inference_out.tif"

    resdepth_state_dict = torch.load(ckpt)

    print("Loaded model from epoch", resdepth_state_dict["epoch"])
    model_args = {
        "n_input_channels": 5,
        "start_kernel": 64,
        "depth": 5,
        "act_fn_encoder": "relu",
        "act_fn_decoder": "relu",
        "act_fn_bottleneck": "relu",
        "up_mode": "transpose",
        "do_BN": True,
        "outer_skip": True,
        "outer_skip_BN": False,
        "bias_conv_layer": True,
    }
    checkpoint_model = UNet(**model_args)

    device = torch.device("cuda")
    checkpoint_model.load_state_dict(
        {k[5:]: v for k, v in resdepth_state_dict["state_dict"].items()},
    )  # cut off the 'unet.' in state dict???
    checkpoint_model.eval()
    checkpoint_model.to(device)

    with torch.no_grad():  # avoid memory issues?

        BATCH_SIZE = 1

        train_transforms = None  # Compose([remove_bbox])
        # VALIDATION
        # val_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only"
        val_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_small_errors_only"
        print("Loading dataset", val_directory)

        input_layers = [
            "dsm",
            "ortho_left",
            "ortho_right",
            "triangulation_error",
            "nodata_mask",
        ]

        dataset = TGDSMOrthoDataset(
            root=val_directory,
            split="train",
            transforms=train_transforms,
            input_layers=input_layers,
        )
        print("Dataset bounds", dataset.bounds)  # torchgeo stores bounds
        minx, miny, maxx, maxy = (
            dataset.bounds.minx,
            dataset.bounds.miny,
            dataset.bounds.maxx,
            dataset.bounds.maxy,
        )

        C = 1  # potentially can include multiple model outputs as bands in 1 raster
        H = math.ceil(maxy - miny + 1)
        W = math.ceil(maxx - minx + 1)

        print("Creating output raster with shape", C, H, W)

        # TODO can switch this to torch tensor if speed an issue (unlikely to start)
        out = np.zeros((C, H, W), dtype=np.float32)

        sampler = GridGeoSampler(
            dataset, size=TILE_SIZE, stride=TILE_SIZE
        )  # keep default units (pixels)

        dataloader = DataLoader(
            dataset, sampler=sampler, collate_fn=stack_samples, num_workers=1  # 20
        )

        val_output_dir = f"{os.path.basename(val_directory)}_{os.path.basename(ckpt)}"
        os.makedirs(val_output_dir, exist_ok=True)
        print("output directory: ", val_output_dir)

        bboxes = (
            []
        )  # Potentially save the image patch bounds for plotting/tracking examples

        # TODO add tqdm
        print("Running inference on each patch in dataset...")
        for b, patch in enumerate(dataloader):
            assert len(patch["bbox"]) == 1, "batch size should be 1"
            bbox = patch["bbox"][0]
            bboxes.extend(
                patch["bbox"]
            )  # how to add this back in? had to remove due to frozen dataclass issue

            inputs = patch["inputs"]
            # Create a new tensor where the inputs have been appropriately normalized
            normalized_inputs = torch.zeros_like(patch["inputs"], device=device)

            if normalization_method == "minmax":

                min_max_per_band = [
                    [0, 3266.77],
                    [0, 2045],
                    [0, 2007],
                    [0, 6.98],
                    [0, 1],
                ]
                for band in range(len(min_max_per_band)):
                    normalized_inputs[:, band] = (
                        inputs[:, band] - min_max_per_band[band][0]
                    ) / min_max_per_band[band][1]

                output = call_model(checkpoint_model, normalized_inputs)
                output = output.cpu()
                output = (output * min_max_per_band[0][1]) + min_max_per_band[0][0]
            elif normalization_method == "meanstd":
                dsm_mean = patch["inputs"][:, 0].nanmean().numpy()

                normalized_inputs[:, 0] = torchvision.transforms.Normalize(
                    dsm_mean, dsm_std
                )(inputs[:, 0])

                normalized_inputs[:, 1:3] = torchvision.transforms.Normalize(
                    ortho_mean, ortho_std
                )(inputs[:, 1:3])

                normalized_inputs[:, 3] = torchvision.transforms.Normalize(
                    trierror_mean, trierror_std
                )(inputs[:, 3])

                # no normalization for the nodata mask layer 3

                output = call_model(checkpoint_model, normalized_inputs)
                output = output.cpu()

                output = (output * dsm_std) + dsm_mean
            else:
                raise NotImplementedError

            x, y = bbox.minx, bbox.miny
            x = int(x - minx)
            y = int(maxy - y)

            # TODO check math for potential off-by-one errors
            # output = patch["target"]  # just pass through GT for testing purposes, expect difference map == 0
            out[0, y - TILE_SIZE : y, x : x + TILE_SIZE] = output.numpy()

        print("Writing output raster:", output_filename)

        RESOLUTION = 1  # meters
        transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)

        with rasterio.open(
            output_filename,
            "w",
            driver="GTiff",
            height=out.shape[1],
            width=out.shape[2],
            count=1,
            dtype=str(out.dtype),
            crs="EPSG:32610",
            transform=transform,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
        ) as new_dataset:
            new_dataset.write(out[0], 1)
        # gdal_opt='-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER'
