"""Evaluate trained model from checkpoint"""
import sys
import os

import matplotlib.pyplot as plt
import torch
import torchvision

from torchgeo.samplers import (
    RandomBatchGeoSampler,
)  # appropriate for tile based dataset
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
)

TILE_SIZE = 256

if __name__ == "__main__":
    input("Hit enter to evaluate with previous checkpoint defined in code")

    # BELOW THIS LINE IS JUST DEMO FIGURES, SHOULD BE SPLIT INTO EVAL SCRIPT OR MODE
    BATCH_SIZE = 4

    train_transforms = Compose([remove_bbox])
    # VALIDATION
    val_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/tile_stacks_prelim_west_half_baker_mapproject_w_asp_refdem"
    print("Loading validation dataset")
    dataset = TGDSMOrthoDataset(
        root=val_directory, split="train", transforms=train_transforms
    )
    sampler = RandomBatchGeoSampler(
        dataset, batch_size=BATCH_SIZE, size=TILE_SIZE, length=100
    )
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=stack_samples, num_workers=20
    )

    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/lightning_logs/version_23/checkpoints/epoch=562-step=2815.ckpt"
    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/lightning_logs/version_24/checkpoints/epoch=999-step=50000.ckpt"

    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/tb_logs/resdepth_torchgeo/version_1/checkpoints/epoch=8-step=2250.ckpt"
    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/checkpoints/epoch=4-step=1250.ckpt"
    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/checkpoints/epoch=14-step=3750.ckpt"
    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/checkpoints/last-v12.ckpt"
    # ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/checkpoints/epoch=55-step=14000.ckpt"
    ckpt = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/tb_logs/resdepth_torchgeo/version_14/checkpoints/epoch=870-step=217750.ckpt"
    val_output_dir = f"{os.path.basename(val_directory)}_{os.path.basename(ckpt)}"
    os.makedirs(val_output_dir, exist_ok=True)
    resdepth_state_dict = torch.load(ckpt)
    # resdepth_state_dict = torch.load(run_folder + "/checkpoints/Model_best.pth") # 2021 trained model
    print("Loaded best model from epoch", resdepth_state_dict["epoch"])
    model_args = {
        "n_input_channels": 3,
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
    # # print(resdepth_state_dict.keys())
    # # print(type(resdepth_state_dict["state_dict"]))
    # print(resdepth_state_dict["state_dict"].keys())

    # TODO if statement
    checkpoint_model.load_state_dict(
        {k[5:]: v for k, v in resdepth_state_dict["state_dict"].items()}
    )  # cut off the 'unet.' in state dict???
    # checkpoint_model.load_state_dict(resdepth_state_dict["model_state_dict"])

    bboxes = []
    for b, batch in enumerate(dataloader):
        print("On batch", b)
        # print(batch)
        # print(batch["bbox"])
        # bboxes.extend(batch["bbox"]) # how to add this back in? had to remove due to frozen dataclass issue

        # nanmean vs mean: the actual problem is holes in the orthoimage inputs???
        dsm_mean = batch["inputs"][:, 0].nanmean().numpy()
        # tile_dsm_normalized = torchvision.transforms.Normalize(dsm_mean, dsm_std)(torch.tensor([tile_dsm.data])).numpy().squeeze()
        normalized_inputs = torch.tensor(batch["inputs"])
        normalized_inputs[:, 0] = torchvision.transforms.Normalize(dsm_mean, dsm_std)(
            normalized_inputs[:, 0]
        )

        normalized_inputs[:, 1:] = torchvision.transforms.Normalize(
            ortho_mean, ortho_std
        )(normalized_inputs[:, 1:])

        output = call_model(checkpoint_model, normalized_inputs)
        # print(output.shape)

        # # Un-normalize the output
        # print("output.shape", output.shape)
        # print("dsm_std", type(dsm_std), dsm_std)
        # print("dsm_mean", type(dsm_mean), dsm_mean)

        # print("Plotting batch")
        print("dsm_mean:", dsm_mean)
        print("dsm_std:", dsm_std)
        print("output mean of batch:", output.mean())
        output = (output * dsm_std) + dsm_mean
        print("output mean of batch:", output.mean())
        print("output range of batch:", output.min(), output.max())

        # Plot example
        image = batch["inputs"]
        target = batch["target"]

        # print(image.shape)
        # print(target.shape)
        fig, ax = plt.subplots(BATCH_SIZE, 8, figsize=(15, 2 * BATCH_SIZE))
        plt.suptitle(
            f"Validation tiles: Batch of size {image.shape[0]} \nCheckpoint: {ckpt}\nDataset: {val_directory}, Tiles=???"
        )
        # print(batch["bbox"])
        # bboxes.extend(batch["bbox"])
        # shapely.geometry.Box()

        # TODO do rest of loop
        plt.rcParams["figure.dpi"] = 300

        for i in range(BATCH_SIZE):
            initial_dem = image[i][0].numpy().squeeze()
            ax[i][0].imshow(hillshade(initial_dem), cmap="gray", rasterized=True)
            im = ax[i][0].imshow(initial_dem, alpha=0.5, rasterized=True)
            ax[i][0].set_title("initial DEM")
            plt.colorbar(im, ax=ax[i][0], fraction=0.04)

            im = ax[i][1].imshow(
                image[i][1].numpy().squeeze(), cmap="gray", rasterized=True
            )
            ax[i][1].set_title("ortho left")
            plt.colorbar(im, ax=ax[i][1], fraction=0.04)

            im = ax[i][2].imshow(
                image[i][2].numpy().squeeze(), cmap="gray", rasterized=True
            )
            ax[i][2].set_title("ortho right")
            plt.colorbar(im, ax=ax[i][2], fraction=0.04)

            gt_dem = target[i].numpy().squeeze()
            ax[i][3].imshow(hillshade(gt_dem), cmap="gray", rasterized=True)
            im = ax[i][3].imshow(gt_dem, alpha=0.5, rasterized=True)
            ax[i][3].set_title("lidar DEM")
            plt.colorbar(im, ax=ax[i][3], fraction=0.04)

            output_dem = output[i]
            # # Normalize the naive way
            # output_dem = (output_dem * dsm_std) + dsm_mean

            ax[i][4].imshow(hillshade(output_dem), cmap="gray", rasterized=True)
            im = ax[i][4].imshow(output_dem, alpha=0.5, rasterized=True)
            ax[i][4].set_title("refined DEM")
            plt.colorbar(im, ax=ax[i][4], fraction=0.04)

            im = ax[i][5].imshow(output_dem - initial_dem, cmap="RdBu", rasterized=True)
            ax[i][5].set_title("residual DEM refinement")
            plt.colorbar(im, ax=ax[i][5], fraction=0.04)

            im = ax[i][6].imshow(initial_dem - gt_dem, cmap="RdBu", rasterized=True)
            ax[i][6].set_title("Initial - Lidar")
            plt.colorbar(im, ax=ax[i][6], fraction=0.04)

            im = ax[i][7].imshow(output_dem - gt_dem, cmap="RdBu", rasterized=True)
            ax[i][7].set_title("Output - Lidar")
            plt.colorbar(im, ax=ax[i][7], fraction=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(val_output_dir, f"val_rd_out_batch_{b}.png"))
        plt.close()

    # Plot all the bboxes???
    # Print each box loaded in this dataset
    for box in sorted(bboxes, key=lambda b1: b1.minx):
        print(box.minx, box.miny)
