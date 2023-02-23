"""Training script for ResDepth model"""
# Usage: python train.py <checkpoint_to_resume_from>
import os
import random
import sys

import numpy as np
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning import Trainer

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

    # Instead try with mosaic of above-treeline tiles
    train_directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_small_errors_only"
    val_directory = (
        "/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only"
    )

    train_transforms = Compose([remove_bbox])

    input_layers = [
        "dsm",
        "ortho_left",
        "ortho_right",
        "triangulation_error",
        "nodata_mask",
    ]

    train_dataset = TGDSMOrthoDataset(
        root=train_directory,
        split="train",
        transforms=train_transforms,
        input_layers=input_layers,
    )
    val_dataset = TGDSMOrthoDataset(
        root=val_directory,
        split="train",
        transforms=train_transforms,
        input_layers=input_layers,
    )

    print(f"Length of dataset: {len(train_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")

    # Set up dataloaders for train & validation datasets

    BATCH_SIZE = 20
    NUM_WORKERS = 20
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

    no_logs = False
    if len(sys.argv) == 2 and sys.argv[1] == "--no-logs":
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
        max_epochs=-1,  # infinite
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )

    # Initialize the model
    checkpoint = None
    checkpoint_fn = None
    if len(sys.argv) == 2 and not no_logs:
        checkpoint_fn = sys.argv[1]
        checkpoint = torch.load(checkpoint_fn)
        print("Want to resume from", checkpoint_fn)

    # TODO specify other parameters as command line args
    from x_unet import XUnet

    class XUnetWithSkipConnection(XUnet):
        """Add Stucker et al residual connection from UNet initial input DSM to the output, so that network learns to compute the residual correction"""

        def forward(self, x):
            residual = super().forward(x)
            x_0 = x[:, 0, :, :]  # initial DSM passed in to the network
            x_0 = x_0.unsqueeze(1)

            return x_0 + residual

    x_unet = XUnetWithSkipConnection
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
        loss_fn=torch.nn.MSELoss,
    )

    # Train model (until interrupted, unless number of epochs is specified)
    trainer.fit(
        model,
        ckpt_path=checkpoint_fn,  # can do this way if resuming (not changing LR)
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("Finished training")
