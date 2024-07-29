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

from pytorch_lightning.cli import LightningCLI

# Import our dataset
# from torchgeo_dataset import TGDSMOrthoDataset
from train_utils import remove_bbox
from resdepth_lightning_module import TGDSMLightningModule
from tgdsm_lightning_data_module import TGDSMDataModule

TILE_SIZE = 256  # input patch height and width, pixels

# TODO Set consistent seeds (want to see variety of inputs right now)
# LightningCLI._set_seed() #no
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

class MyLightningCLI(LightningCLI):
    pass

    # TODO how to link up model input layers with dataset input layers
    # def add_arguments_to_parser(self, parser):
    #     parser.link_arguments("data.batch_size", "model.batch_size")


def cli_main():
    """
    Set up configurable training with command-line arguments
    """
    # TODO callbacks for checkpoints, logging, etc.
    cli = MyLightningCLI(
        model_class=TGDSMLightningModule,
        datamodule_class=TGDSMDataModule,
        seed_everything_default=1,
        subclass_mode_data=True,
        subclass_mode_model=True
    )

if __name__ == "__main__":
    cli_main()
