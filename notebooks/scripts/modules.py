# misc imports
from typing import Dict

# The DeepDEM framework uses a UNet for DEM refinement
from UNet import UNet

# pytorch imports
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2


# pytorch-lightning imports
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import lightning as L

# torchgeo imports
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.datamodules import GeoDataModule
from torchgeo.datamodules.utils import MisconfigurationException

import torchvision.transforms.functional as TF
import random

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles, p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        if self.p > random.random():
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)
        
        return x


class ImageAugmentation():
    """Custom class implementing data augmentation for use in DeepDEM"""
    
    def __init__(self, 
                 hflip=True, hflip_p=0.5,
                 vflip=True, vflip_p=0.5,
                 rotation=True, rotation_p=0.5, 
                 rotation_angles = (0, 90, 180, 270)) -> None:
        
        self.hflip = hflip
        self.hflip_p = hflip_p

        self.vflip = vflip
        self.vflip_p = vflip_p

        self.rotation = rotation
        self.rotation_p = rotation_p
        self.rotation_angles = rotation_angles

        self.compose_transform()

    def compose_transform(self):
        """Compose transformation"""

        transforms = []
        if self.hflip:
            transforms.append(v2.RandomHorizontalFlip(p=self.hflip_p))

        if self.vflip:
            transforms.append(v2.RandomVerticalFlip(p=self.vflip_p))

        if self.rotation:
            transforms.append(RotationTransform(self.rotation_angles, self.rotation_p))

        self.transform = v2.Compose(transforms=transforms)

    def __call__(self, x):
        return self.transform(x)


class InputRasterData(RasterDataset):
    """
    Torchgeo wrapper for training data
    """
    is_image=True
    separate_files=True
    filename_glob = "final*DEM*.tif"
    all_bands = ["stereo_DEM_filled", "ortho_left", "ortho_right", "triangulation_errors", "nodata_mask"]

class LabelRasterData(RasterDataset):
    """
    Torchgeo wrapper for training labels
    """
    is_image=False
    separate_files=True
    filename_glob = "final*lidar*.tif"
    all_bands = ["DEM"]

class CustomGeoDataModule(GeoDataModule):
    """
    Custom GeoDataModule that can be used with Lightning DataLoaders
    """
    generator = torch.Generator().manual_seed(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sampler = kwargs['sampler']

        self.train_datatset = kwargs['train_dataset']
        self.train_sampler = kwargs['train_sampler']

        self.test_datatset = kwargs['test_dataset']
        self.test_sampler = kwargs['test_sampler']

        self.val_datatset = kwargs['val_dataset']
        self.val_sampler = kwargs['val_sampler']

        self.kwargs = kwargs
        
    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = self.kwargs['train_dataset']
            self.train_batch_sampler = self.kwargs['train_sampler']
        
        if stage in ["validate"]:
            self.val_dataset = self.kwargs['val_dataset']
            self.val_sampler = self.kwargs['val_sampler']
        if stage in ["test"]:
            self.test_dataset = self.kwargs['test_dataset']
            self.test_sampler = self.kwargs['test_sampler']



    def train_dataloader(self):
        """Implement PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'.
        """
        dataset = self.train_dataset
        sampler = self.train_sampler
        if (dataset is not None) and (sampler is not None):
            batch_size = self.kwargs["batch_size"]
            
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
                # batch_sampler=RandomBatchGeoSampler, # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)
        
    def val_dataloader(self):
        """Implement PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'.
        """
        dataset = self.val_dataset
        sampler = self.val_sampler
        if (dataset is not None) and (sampler is not None):
            batch_size = self.kwargs["batch_size"]
            
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
                # batch_sampler=RandomBatchGeoSampler, # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)
        
    def test_dataloader(self):
        """Implement PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying test samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'.
        """
        dataset = self.test_dataset
        sampler = self.test_sampler
        if (dataset is not None) and (sampler is not None):
            batch_size = self.kwargs["batch_size"]
            
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
                batch_sampler=RandomBatchGeoSampler, # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)

    def predict_dataloader(self):
        """Implement PyTorch DataLoaders for predict.

        Returns:
            A collection of data loaders specifying predicting samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'predict_dataset'.
        """
        dataset = self.predict_dataset
        sampler = self.predict_sampler
        batch_sampler = self.predict_sampler
        if (dataset is not None) and (sampler is not None):
            batch_size = self.kwargs["batch_size"]
            
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
                # batch_sampler=batch_sampler, # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)

    def transfer_batch_to_device(
        self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Tensor]:
        
        # del batch["crs"]
        # del batch["bbox"]

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

class DeepDEMModel(L.LightningModule):
    """Derived class of lightning module that will be use to train the DeepDEM model"""
    def __init__(self, n_input_channels):
        super().__init__()
        self.model = UNet(n_input_channels=n_input_channels)
        self.loss = nn.functional.l1_loss

    def training_step(self, *args, **kwargs)->None:
        # Training step
        x, y = args[0]['image'], args[0]['mask']
        z = self.model(x)

        loss = self.loss(z, y)
        self.log("Training loss: ", loss)

    def test_step(self, *args, **kwargs):
        # Test step
        x, y = args[0]['image'], args[0]['mask']
        z = self.model(x)
        test_loss = self.loss(z, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configuring optimizers"""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer