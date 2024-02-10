# misc imports
from pathlib import Path
from typing import Dict

# The DeepDEM framework uses a UNet for DEM refinement
from UNet import UNet

# pytorch imports
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
import torch

# pytorch-lightning imports
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import lightning as L

# torchgeo imports
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, RandomBatchGeoSampler
from torchgeo.datamodules import GeoDataModule
from torchgeo.datamodules.utils import MisconfigurationException

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

if __name__ == '__main__':

    # specify paths of input dataset and the labels
    input_dataset = InputRasterData('/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/baker_csm_stack/cropped_rasters/input_channels')
    label_dataset = LabelRasterData('/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/baker_csm_stack/cropped_rasters/target_labels')

    # RasterDatasets support '&' operations for merging
    mtbaker_dataset = input_dataset & label_dataset

    # Sampler
    this_sampler = RandomGeoSampler(mtbaker_dataset, size=256, length=1000)

    datamodule_params = {
        'dataset_class': RasterDataset,
        'batch_size': 16, 
        'patch_size': 256,
        'num_workers':8,
        'dataset':mtbaker_dataset,
        'sampler':this_sampler,
        
        'train_dataset':mtbaker_dataset,
        'val_dataset':mtbaker_dataset,
        'test_dataset':mtbaker_dataset,
        
        'train_sampler':this_sampler,
        'test_sampler':this_sampler,
        'val_sampler':this_sampler,

        'collate_fn':stack_samples
    }
    
    datamodule = CustomGeoDataModule(**datamodule_params)
    
    model = DeepDEMModel(n_input_channels=5)

    checkpoint_directory = Path('./checkpoint_directory')
    checkpoint_directory.mkdir(exist_ok=True)

    trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto",
                         default_root_dir=checkpoint_directory, limit_train_batches=100,
                         max_epochs=500)

    trainer.fit(model=model, train_dataloaders=datamodule)