"""Evaluate trained model from checkpoint"""
import sys
import os

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import torch
import torchvision

from torchgeo.samplers import GridGeoSampler  # appropriate for tile based dataset
from torchgeo.datasets import stack_samples

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# TODO figure out how to change to relative import of ResDepth UNet
sys.path.insert(0, "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/ResDepth/")
from lib.UNet import UNet

# Import our dataset
from torchgeo_dataset import TGDSMOrthoDataset

# Imports from train.py
from train_utils import (
    remove_bbox,
    dsm_std,
    ortho_mean,
    ortho_std,
    hillshade,
    call_model,
    trierror_mean,
    trierror_std,
)

TILE_SIZE = 256

if __name__ == "__main__":
    BATCH_SIZE = 1

    train_transforms = None  # Compose([remove_bbox])

    # VALIDATION
    # val_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VALIDATION_tile_stack_baker_128_global_coreg"
    val_directory = (
        "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only"
    )

    print("Loading validation dataset")
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

    sampler = GridGeoSampler(
        dataset, size=TILE_SIZE, stride=TILE_SIZE
    )  # keep default units pixels

    dataloader = DataLoader(
        dataset, sampler=sampler, collate_fn=stack_samples, num_workers=20
    )

    # TODO change to better CLI arguments handling
    ckpt = sys.argv[1]
    normalization_method = sys.argv[2]

    # Output directory for figures
    val_output_dir = f"{os.path.basename(val_directory)}_{os.path.basename(ckpt)}"
    os.makedirs(val_output_dir, exist_ok=True)

    print("output directory: ", val_output_dir)
    resdepth_state_dict = torch.load(ckpt)
    print("Loaded best model from epoch", resdepth_state_dict["epoch"])

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

    checkpoint_model.load_state_dict(
        {k[5:]: v for k, v in resdepth_state_dict["state_dict"].items()}
    )  # cut off the 'unet.' in state dict???

    bboxes = []
    num_examples = 0
    num_to_evaluate = 20
    limit = 20

    # 
    device = torch.device("cuda")
    checkpoint_model.eval()

    with torch.no_grad():
        for b, patch in enumerate(dataloader):
            print("On patch", b)
            print(f"{b} {patch['bbox']}")
            num_examples += 1
            if num_examples > num_to_evaluate:
                break
            print(f"inputs shape = {patch['inputs'].shape}")

            x = patch["inputs"]
            normalized_inputs = torch.zeros_like(x)

            if normalization_method == "minmax":

                min_max_per_band = [[0, 3266.77], [0, 2045], [0, 2007], [0, 6.98], [0, 1]]
                for band in range(len(min_max_per_band)):
                    normalized_inputs[:, band] = (
                        x[:, band] - min_max_per_band[band][0]
                    ) / min_max_per_band[band][1]

                output = call_model(checkpoint_model, normalized_inputs)
                output = (output * min_max_per_band[0][1]) + min_max_per_band[0][0]

            elif normalization_method == "meanstd":
                # nanmean vs mean: the actual problem is holes in the orthoimage inputs???
                dsm_mean = patch["inputs"][:, 0].nanmean().numpy()
                # tile_dsm_normalized = torchvision.transforms.Normalize(dsm_mean, dsm_std)(torch.tensor([tile_dsm.data])).numpy().squeeze()
                normalized_inputs[:, 0] = torchvision.transforms.Normalize(
                    dsm_mean, dsm_std
                )(x[:, 0])

                normalized_inputs[:, 1:3] = torchvision.transforms.Normalize(
                    ortho_mean, ortho_std
                )(x[:, 1:3])

                normalized_inputs[:, 3] = torchvision.transforms.Normalize(
                    trierror_mean, trierror_std
                )(x[:, 3])

                # Nodata mask gets passed through
                normalized_inputs[:, 4] = x[:, 4]

                output = call_model(checkpoint_model, normalized_inputs)
                print(output.shape)

                # # Un-normalize the output
                output = (output * dsm_std) + dsm_mean

                print("output mean of patch:", output.mean())
                print("output range of patch:", output.min(), output.max())
            else:
                raise NotImplementedError

            # Plot example
            image = patch["inputs"]
            target = patch["target"]

            nrows = 3
            ncols = 4
            fig, ax = plt.subplots(3, 4, figsize=(ncols * 4, nrows * 4 + 2), squeeze=False)
            plt.suptitle(
                f"Validation tiles: patch of size {image.shape[0]} \nCheckpoint: {ckpt}\nDataset: {val_directory}"  # , Tiles=???"
            )

            # TODO do rest of loop
            plt.rcParams["figure.dpi"] = 300

            ax = ax.reshape((1, -1))

            for i in range(BATCH_SIZE):
                for j in range(ax.shape[1]):
                    ax[i][j].set_xticks([])
                    ax[i][j].set_yticks([])
                    # ax.set_aspect(1)

                # Define the range so that DEMs have a consistent colorscale
                initial_dem = image[i][0].numpy().squeeze()
                gt_dem = target[i].numpy().squeeze()

                output_dem = output.squeeze()

                # Set range for consistent DEM colorscales
                DEM_VMIN = min(gt_dem.min(), initial_dem.min())  # , output_dem.min())
                DEM_VMAX = max(gt_dem.max(), initial_dem.max())  # , output_dem.max())

                # Set range for difference maps
                DIFF_VMAX = 2
                DIFF_VMIN = -1 * DIFF_VMAX

                # Show orthoimages with original contrast
                im = ax[i][0].imshow(
                    image[i][1].numpy().squeeze(), cmap="gray", rasterized=True
                )
                ax[i][0].set_title("ortho nadir")
                plt.colorbar(im, ax=ax[i][0], fraction=0.04)

                # Include scalebar
                scale = 1  # meter
                ax[i][0].add_artist(ScaleBar(scale, fixed_value=50))

                im = ax[i][1].imshow(
                    image[i][2].numpy().squeeze(), cmap="gray", rasterized=True
                )
                ax[i][1].set_title("ortho off-nadir")
                plt.colorbar(im, ax=ax[i][1], fraction=0.04)

                # Show shaded relief maps (color hillshades) of input & lidar DEMs

                ax[i][2].imshow(hillshade(initial_dem), cmap="gray", rasterized=True)
                im = ax[i][2].imshow(
                    initial_dem, alpha=0.5, rasterized=True, vmin=DEM_VMIN, vmax=DEM_VMAX
                )
                ax[i][2].set_title("initial DEM")
                plt.colorbar(im, ax=ax[i][2], fraction=0.04)

                ax[i][3].imshow(hillshade(gt_dem), cmap="gray", rasterized=True)
                im = ax[i][3].imshow(
                    gt_dem, alpha=0.5, rasterized=True, vmin=DEM_VMIN, vmax=DEM_VMAX
                )
                ax[i][3].set_title("lidar DEM")
                plt.colorbar(im, ax=ax[i][3], fraction=0.04)

                refinement = output_dem - initial_dem
                refinement_vmax = abs(refinement).max()
                refinement_vmin = -1 * refinement_vmax
                im = ax[i][4].imshow(
                    output_dem - initial_dem,
                    cmap="RdBu",
                    rasterized=True,
                    vmin=refinement_vmin,
                    vmax=refinement_vmax,
                )
                ax[i][4].set_title("residual DEM refinement")
                plt.colorbar(im, ax=ax[i][4], fraction=0.04)

                diff_initial_gt = initial_dem - gt_dem
                im = ax[i][5].imshow(
                    diff_initial_gt,
                    cmap="RdBu",
                    vmin=DIFF_VMIN,
                    vmax=DIFF_VMAX,
                    rasterized=True,
                )
                ax[i][5].set_title("Initial - Lidar")
                plt.colorbar(im, ax=ax[i][5], fraction=0.04)

                diff_output_gt = output_dem - gt_dem
                im = ax[i][6].imshow(
                    diff_output_gt,
                    cmap="RdBu",
                    vmin=DIFF_VMIN,
                    vmax=DIFF_VMAX,
                    rasterized=True,
                )
                ax[i][6].set_title("Output - Lidar")
                plt.colorbar(im, ax=ax[i][6], fraction=0.04)

                # Show refined output
                ax[i][7].imshow(hillshade(output_dem), cmap="gray", rasterized=True)
                im = ax[i][7].imshow(
                    output_dem, alpha=0.5, rasterized=True, vmin=DEM_VMIN, vmax=DEM_VMAX
                )
                ax[i][7].set_title("refined DEM")
                plt.colorbar(im, ax=ax[i][7], fraction=0.04)

                # Show the nodata mask
                im = ax[i][8].imshow(
                    image[i][4].numpy().squeeze(), cmap="gray", rasterized=True
                )  # vmin=0, vmax=1)
                ax[i][8].set_title("Nodata mask")
                plt.colorbar(im, ax=ax[i][8], fraction=0.04)

                # Show the nodata mask
                im = ax[i][9].imshow(
                    image[i][3].numpy().squeeze(), cmap="inferno", rasterized=True
                )  # vmin=0, vmax=1)
                ax[i][9].set_title("Triangulation error")
                plt.colorbar(im, ax=ax[i][9], fraction=0.04)

                # Histograms
                # TODO make these look nice and add axis labels back
                import numpy as np

                HIST_VMIN = DIFF_VMIN  # min(diff_output_gt.min(), diff_initial_gt.min())
                HIST_VMAX = DIFF_VMAX  # max(diff_output_gt.max(), diff_initial_gt.max())
                ax_hist = ax[i][10]
                ax_hist.set_xticks(np.arange(HIST_VMIN, HIST_VMAX))  # , step=0.1))
                ax_hist.axvline(x=0, color="k", linewidth=0.5, linestyle=":")
                ax_hist.hist(
                    (diff_initial_gt).reshape(-1),
                    bins=250,
                    color="red",
                    label="Before",
                    alpha=0.3,
                    range=(HIST_VMIN, HIST_VMAX),
                )
                ax_hist.hist(
                    (output_dem - gt_dem).reshape(-1),
                    bins=250,
                    color="blue",
                    label="After",
                    alpha=0.3,
                    range=(HIST_VMIN, HIST_VMAX),
                )
                ax_hist.set_xlabel("Elev. Diff. (m)")
                ax_hist.set_ylabel("Count (px)")
                ax_hist.text(
                    0.05,
                    0.95,
                    "Initial",
                    va="top",
                    color="red",
                    transform=ax_hist.transAxes,
                    fontsize=16,
                )
                ax_hist.text(
                    0.05,
                    0.85,
                    "Refined",
                    va="top",
                    color="blue",
                    transform=ax_hist.transAxes,
                    fontsize=16,
                )

                # 12th figure?
                range_str = f"gt range [{gt_dem.min():.1f}, {gt_dem.max():.1f}]\noutput range [{output_dem.min():.1f},{output_dem.max():.1f}]"
                ax[i][11].set_title(range_str)
                bb = patch["bbox"][0]
                bbox_str = f"minx={bb.minx:.0f}, miny={bb.miny:.0f}"
                ax[i][11].text(
                    0.05,
                    0.95,
                    bbox_str,
                    va="top",
                    color="black",
                    transform=ax[i][11].transAxes,
                    fontsize=12,
                )

                l1_loss = abs(initial_dem - gt_dem).mean()
                loss_str = f"L1 loss: {l1_loss:.4f}"
                ax[i][11].text(
                    0.05,
                    0.85,
                    loss_str,
                    va="top",
                    color="red",
                    transform=ax[i][11].transAxes,
                    fontsize=12,
                )
                l2_loss = ((initial_dem - gt_dem) ** 2).mean()
                l2_loss_str = f"L2 loss: {l2_loss:.4f}"
                ax[i][11].text(
                    0.05,
                    0.75,
                    l2_loss_str,
                    va="top",
                    color="red",
                    transform=ax[i][11].transAxes,
                    fontsize=12,
                )

                # TODO change variable names
                l1_loss = abs(output_dem - gt_dem).mean()
                loss_str = f"L1 loss: {l1_loss:.4f}"
                ax[i][11].text(
                    0.05,
                    0.65,
                    loss_str,
                    va="top",
                    color="blue",
                    transform=ax[i][11].transAxes,
                    fontsize=12,
                )
                l2_loss = ((output_dem - gt_dem) ** 2).mean()
                l2_loss_str = f"L2 loss: {l2_loss:.4f}"
                ax[i][11].text(
                    0.05,
                    0.55,
                    l2_loss_str,
                    va="top",
                    color="blue",
                    transform=ax[i][11].transAxes,
                    fontsize=12,
                )

            plt.tight_layout()
            plt.savefig(os.path.join(val_output_dir, f"val_rd_out_patch_{b}.png"))
            plt.close()

        # Plot all the bboxes???
        # Print each box loaded in this dataset
        for box in sorted(bboxes, key=lambda b1: b1.minx):
            print(box.minx, box.miny)
