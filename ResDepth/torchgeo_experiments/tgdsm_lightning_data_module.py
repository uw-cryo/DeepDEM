"""
Define a LightningDataModule for use with the PyTorch LightningCLI.

This class should take in the dataset parameters:
* location of train & validation (& test) directories
* input layers
* desired dataset size, dimensions, etc.
and return dataloaders needed to run the desired action (fit/validate/predict/test/tune)
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchgeo.samplers import (
    RandomBatchGeoSampler,
)  # appropriate for tile based dataset
from torchgeo.datasets import stack_samples

from torchvision.transforms import Compose

from torchgeo_dataset import TGDSMOrthoDataset
from train_utils import remove_bbox


class TGDSMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_directory="/mnt/1.0_TB_VOLUME/sethv/shashank_data/TRAIN_tile_stack_baker_small_errors_only",
        val_directory="/mnt/1.0_TB_VOLUME/sethv/shashank_data/VAL_tile_stack_baker_small_errors_only",
        input_layers=[
            "dsm",
            "ortho_left",
            "ortho_right",
            "triangulation_error",
            "nodata_mask",
        ],
        # transforms=[remove_bbox],
        batch_size=20,
        tile_size=256,  # 256x256 patches
        train_length=5000,
        val_length=500,
        num_workers=20,
        dataset="baker2015_singletile"
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define dataset location
        self.train_directory = train_directory
        self.val_directory = val_directory

        # Input layers to load
        self.input_layers = input_layers

        # Same transforms for train + val
        transforms = []  # TODO would like to support dataset transforms but can't pass through CLI
        self.train_transforms = Compose(transforms + [remove_bbox])
        self.val_transforms = Compose(transforms + [remove_bbox])

        self.batch_size = batch_size
        self.tile_size = tile_size
        self.train_length = train_length
        self.val_length = val_length
        self.num_workers = num_workers
        self.dataset = dataset

    def prepare_data(self):
        # Data is already downloaded
        pass

    def setup(self, stage: str):
        self.train_dataset = TGDSMOrthoDataset(
            root=self.train_directory,
            split="train",
            dataset=self.dataset,
            transforms=self.train_transforms,
            input_layers=self.input_layers,
        )
        self.val_dataset = TGDSMOrthoDataset(
            root=self.val_directory,
            split="val",
            dataset=self.dataset,
            transforms=self.val_transforms,
            input_layers=self.input_layers,
        )

        # TODO setup test & predict datasets with no ground truth DEM

    def train_dataloader(self):
        train_sampler = RandomBatchGeoSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            size=self.tile_size,
            length=self.train_length,
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):
        val_sampler = RandomBatchGeoSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            size=self.tile_size,
            length=self.train_length,
        )

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        raise NotImplementedError("test dataloader is not implemented yet")

    def predict_dataloader(self):
        raise NotImplementedError("predict dataloader is not implemented yet")
