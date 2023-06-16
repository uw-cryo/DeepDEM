"""Training script for ResDepth model"""
# Usage: python train.py <checkpoint_to_resume_from>
import os
import random
import sys
import glob

import numpy as np
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning import Trainer

from pytorch_lightning.profilers import SimpleProfiler

from torchgeo.samplers import (
    RandomBatchGeoSampler,
)  # appropriate for tile based dataset
from torchgeo.datasets import stack_samples

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# Import our dataset
from torchgeo_dataset import TGDSMOrthoDataset
from train_utils import remove_bbox
from resdepth_lightning_module import TGDSMLightningModule
from xunet_with_skip_connection import XUnetWithSkipConnection

TILE_SIZE = 256  # input patch height and width, pixels

# TODO Set consistent seeds (want to see variety of inputs right now)
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


if __name__ == "__main__":
    # Load dataset from this folder
    train_directory = (
        "/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_128_global_coreg"
    )
    val_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VALIDATION_tile_stack_baker_128_global_coreg/"
    initial_dem_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM_holes_filled_snow_median_subtracted.tif"

    # # Instead try with mosaic of above-treeline tiles
    # train_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_small_errors_only"
    # val_directory = (
    #     "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only"
    # )


    # # TODO fix nan issues
    # initial_dem_root = None
    # train_directory = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK"
    # val_directory = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK"

    train_transforms = Compose([remove_bbox])  # can't keep a "frozen dataclass"

    input_layers = [
        "dsm",
        "ortho_left",
        "ortho_right",
        "triangulation_error",
        "nodata_mask",
    ]

    # train_dataset = TGDSMOrthoDataset(
    #     root=train_directory,
    #     split="train",
    #     transforms=train_transforms,
    #     input_layers=input_layers,
    #     initial_dem_root=initial_dem_root
    # )

    dataset_SCG = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK", split="train", dataset="scg2019", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)
    dataset_Baker2015 = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/baker_csm/baker_csm_stack", split="train", dataset="baker2015_singletile", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)

    # TODO fix but want to do the temporal adjusted input DEM training run w X-UNet to see if learning from scratch without temporal offset is better

    # dataset_Baker2015_small_errors_only_adjusted = TGDSMOrthoDataset(
    #     "/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_small_errors_only",
    #     split="train",
    #     dataset="baker2015_melt_adjusted",
    #     input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms
    # )

    # train_dataset = dataset_SCG | dataset_Baker2015
    # train_dataset = dataset_Baker2015_small_errors_only_adjusted # | dataset_SCG # v123 both, v124 baker60
    train_dataset = dataset_Baker2015 | dataset_SCG
    print("Loaded torchgeo union dataset")

    val_dataset_baker_single_tile = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/baker_csm/baker_csm_stack", split="train", dataset="baker2015_singletile", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)
    val_dataset_SCG = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK", split="train", dataset="scg2019", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)
    val_dataset = val_dataset_baker_single_tile | val_dataset_SCG

    # val_dataset = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK", split="train", dataset="scg2019", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)

    # TODO this was the usual val dataset
    # val_dataset = TGDSMOrthoDataset(
    #     "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only",
    #     split="val",
    #     dataset="baker2015_melt_adjusted",
    #     input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms
    # )

    # val_dataset = TGDSMOrthoDataset(
    #     root=val_directory,
    #     split="train",
    #     transforms=train_transforms,
    #     input_layers=input_layers,
    #     initial_dem_root=initial_dem_root
    #     # initial_dem_root="WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM_holes_filled_snow_median_subtracted.tif"
    # )

    train_dirs = list(glob.glob("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/baker_csm/baker_csm_stack_train_112_tiles/*"))
    train_dataset = TGDSMOrthoDataset(train_dirs, split="train", dataset="baker2015_singletile", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)

    val_dirs = list(glob.glob("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/baker_csm/baker_csm_stack_validation_16_tiles/*"))
    val_dataset = TGDSMOrthoDataset(val_dirs, split="val", dataset="baker2015_singletile", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1, transforms=train_transforms)

    # scg_csm_directory = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/scg_csm/scg_csm_stack"
    # scg_csm_dataset = "scg2019_csm"
    # train_dataset = TGDSMOrthoDataset(
    #     root=scg_csm_directory,
    #     dataset=scg_csm_dataset,
    #     split="train",
    #     transforms=train_transforms,
    #     input_layers=input_layers,
    #     PATCH_SIZE=TILE_SIZE,
    #     initial_dem_root=initial_dem_root
    # )
    # #TODO split up
    # val_dataset = TGDSMOrthoDataset(
    #     root=scg_csm_directory,
    #     dataset=scg_csm_dataset,
    #     split="train",
    #     transforms=train_transforms,
    #     input_layers=input_layers,
    #     PATCH_SIZE=TILE_SIZE,
    #     initial_dem_root=initial_dem_root
    # )

    print(f"Length of dataset: {len(train_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")
    # input("Paused, hit enter to continue:")

    # Set up dataloaders for train & validation datasets

    BATCH_SIZE = 20 # unet =20, x-unet=2
    NUM_WORKERS = 20  # TODO change to max = number of cores available
    train_sampler = RandomBatchGeoSampler(
        train_dataset, batch_size=BATCH_SIZE, size=TILE_SIZE, length=5000
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=stack_samples,
        num_workers=NUM_WORKERS,
    )

    # TODO want gridgeosampler but wasn't working?
    val_sampler = RandomBatchGeoSampler(
        val_dataset, batch_size=BATCH_SIZE, size=TILE_SIZE, length=500
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=stack_samples,
        num_workers=NUM_WORKERS,
    )

    # Set up experiment tracking
    # TODO make more complete log of hyperparameters, which dataset used, etc.
    logger = TensorBoardLogger("tb_logs", name="resdepth_torchgeo")

    checkpoints_dir = os.path.join(logger.log_dir, "checkpoints")

    # Save the 5 best models
    best_checkpoints_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoints_dir,
        save_top_k=5,
        save_last=True,
    )

    # Also save checkpoints at regular intervals
    regular_checkpoints_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoints_dir,
        every_n_epochs=25,
    )

    device_stats = DeviceStatsMonitor()

    # Add profiler to see where code is slow
    profiler = SimpleProfiler(filename="lightning_simple_profiler")
    

    no_logs = False
    if len(sys.argv) >= 2 and sys.argv[1] == "--no-logs":
        no_logs = True

    if no_logs:
        # dry runs for testing
        logger = None
        callbacks = None
    else:
        callbacks = [
            best_checkpoints_callback,
            regular_checkpoints_callback,
            device_stats,
        ]


    trainer = Trainer(
        # resume_from_checkpoint=checkpoint_fn, # TODO doesn't work
        # max_epochs=1, # just to test profiler
        max_epochs=-1,  # infinite
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler
    )
    # Initialize the model
    checkpoint = None
    checkpoint_fn = None
    # if len(sys.argv) == 2 and not no_logs:
    if len(sys.argv) >= 2:
        checkpoint_fn = sys.argv[-1]
        checkpoint = torch.load(checkpoint_fn)
        print(f"Resuming training from checkpoint: {checkpoint_fn}")


    x_unet_args = dict(
        dim=64,
        channels=5,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
        nested_unet_depths=(7, 4, 2, 1),  # nested unet depths, from unet-squared paper
        consolidate_upsample_fmaps=True,  # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
    )

    # model = TGDSMLightningModule(n_input_channels=len(input_layers), lr=0.00002)
    model = TGDSMLightningModule(
        n_input_channels=len(input_layers),
        normalization="meanstd",
        # loss_fn=torch.nn.L1Loss, # v87 mseloss, v90 L1 loss
        loss_fn_module_name="torch.nn",
        loss_fn_class_name="L1Loss",
        lr=0.00005, # v94=0.0005 to see how fast it goes,v96 finetune
        # model=XUnetWithSkipConnection,
        # model_args=x_unet_args
        use_input_dem_mask_for_computing_loss=False #True
    )

    # Train model (until interrupted, unless number of epochs is specified)
    trainer.fit(
        model,
        # TODO temporarily commented out to try resuming a run
        ckpt_path=checkpoint_fn,  # can do this way if resuming (not changing LR)
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("Finished training")
