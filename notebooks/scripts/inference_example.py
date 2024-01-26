"""
Evaluate trained model from checkpoint and produce output DEM + hillshade.
Code needed for minimum working inference example notebook
"""
import sys
import os
import subprocess
from pathlib import Path

import numpy as np
import math

from tqdm import tqdm

import torch
import torchvision

from torchgeo.samplers import GridGeoSampler  # appropriate for tile based dataset
from torchgeo.datasets import stack_samples

from torch.utils.data import DataLoader

import rasterio
from rasterio.transform import from_origin

from UNet import UNet

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

TILE_SIZE = 2048  # may need to be as small as 256x256 depending on model

def main(argv):

    output_filename = "inference_out.tif"
    normalization_method = 'meanstd'

    # # We will be using a GPU for inference
    device = torch.device("cuda")

    # # Define the UNet model parameters
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
    
    assert len(argv)>1, "Provide absolute path to the model weights as an argument"
    ckpt = Path(argv[1])
    assert ckpt.exists(), f"Weights path ({ckpt}) does not exist!"

    resdepth_state_dict = torch.load(ckpt, map_location=device)
    checkpoint_model = UNet(**model_args)
    
    checkpoint_model.load_state_dict(
        {k[5:]: v for k, v in resdepth_state_dict["state_dict"].items()},
    ) 

    # Put model in evaluation mode 
    checkpoint_model.eval()

    # Load model to GPU memory
    checkpoint_model.to(device)

    # Context manager to avoid calculating gradients during inference
    with torch.no_grad():

        BATCH_SIZE = 1

        input_layers = [
            "dsm",
            "ortho_left",
            "ortho_right",
            "triangulation_error",
            "nodata_mask",
        ]

        val_directory = "/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/scg_csm_stack"
        assert Path(val_directory).exists(), "Data path does not exist!"
        
        dataset = "scg2019_csm"

        initial_dem_root = None

        print("Loading dataset(s)", val_directory)

        dataset = TGDSMOrthoDataset(
            root=val_directory,
            dataset=dataset,
            split="test",
            transforms=None,
            input_layers=input_layers,
            PATCH_SIZE=TILE_SIZE,
            initial_dem_root=initial_dem_root
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
            # dataset, size=TILE_SIZE, stride=
            # manually setting stride to work for scg2019_csm
            # kv 01/25/24
            dataset, size=TILE_SIZE, stride=(1371, 1723) 
        )  # keep default units (pixels)

        dataloader = DataLoader(
            dataset, sampler=sampler, collate_fn=stack_samples, num_workers=1  # 20
        )

        if isinstance(val_directory, str):
            val_output_dir = (
                f"{os.path.basename(val_directory)}_{os.path.basename(ckpt)}"
            )
        elif isinstance(val_directory, list):
            basenames = [os.path.basename(d) for d in val_directory]
            concatenated_dirs = "+".join(basenames)
            val_output_dir = f"{concatenated_dirs}_{os.path.basename(ckpt)}"
        else:
            print("Val_directory type=", type(val_directory))
            raise NotImplementedError

        os.makedirs(val_output_dir, exist_ok=True)
        print("output directory: ", val_output_dir)

        bboxes = (
            []
        )  # Potentially save the image patch bounds for plotting/tracking examples

        # TODO add tqdm
        print("Running inference on each patch in dataset...")
        for b, patch in enumerate(tqdm(dataloader)):
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

                # no normalization for the nodata mask layer 4
                normalized_inputs[:, 4] = patch["inputs"][:, 4]

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
            # print(f"Write y1 = {y-TILE_SIZE}, y2={y}, x1={x}, x2={x+TILE_SIZE}")
            print(f"Writing tile ({x}, {y}) output: median={output.median()}, min={output.min()}, max={output.max()})")
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

        # Also write a hillshade for convenience
        output_hs_filename = "hs_" + output_filename
        subprocess.run(["gdaldem", "hillshade", "-compute_edges", output_filename, output_hs_filename])

if __name__ == '__main__':
    main(sys.argv)

