"""Training script for ResDepth model"""
import os
import random

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
    directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/tile_stacks_prelim_west_half_baker_mapproject_w_asp_refdem_11OCT2022"

    train_transforms = Compose([remove_bbox])

    input_layers = [
        "dsm",
        "ortho_left",
        "ortho_right",
        "nodata_mask",
    ]  # skip "nodata_mask"

    dataset = TGDSMOrthoDataset(
        root=directory,
        split="train",
        transforms=train_transforms,
        input_layers=input_layers,
    )

    print(f"Length of dataset: {len(dataset)}")

    # Set up dataloader

    BATCH_SIZE = 20
    sampler = RandomBatchGeoSampler(
        dataset, batch_size=BATCH_SIZE, size=TILE_SIZE, length=5000
    )
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=stack_samples, num_workers=20
    )

    # Set up experiment tracking

    # TODO make more complete log of hyperparameters, which dataset used, etc.
    logger = TensorBoardLogger("tb_logs", name="resdepth_torchgeo")

    checkpoints_dir = os.path.join(logger.log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        monitor="loss",  # TODO use val_loss
        dirpath=checkpoints_dir,
        save_top_k=5,
        save_last=True,
    )

    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, device_stats],
    )

    # Initialize the model
    model = TGDSMLightningModule(n_input_channels=len(input_layers))

    # Train model
    # TODO add validation code!
    trainer.fit(model, train_dataloaders=dataloader)

    print("Finished training")
