from typing import Iterable, Sequence
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, BoundingBox, stack_samples
from torchgeo.samplers import RandomBatchGeoSampler

import kornia as K
from kornia.enhance import normalize

from lightning.pytorch import LightningDataModule

import rasterio
import numpy as np
import traceback

class CustomInputDataset(RasterDataset):
    """Custom dataset that looks for
    1. ASP derived DSM from stereo imagery
    2. Stereo imagery (2 files) used to derive initial DSM estimate
    3. NDVI values coinciding with the extent of the stereo imagery
    4. Triangulation errors associated with DSM generation from ASP
    5. Mask of no-data values where ASP is unable to generate an elevation estimate
    6. Ground truth lidar data to train the model
    """

    is_image = True
    separate_files = True
    filename_regex = r"final_(?P<band>[a-z_]+)\.tif"
    all_bands = [
        "asp_dsm",
        "ortho_left",
        "ortho_right",
        "ndvi",
        "nodata_mask",
        "triangulation_error",
        "lidar_data",
    ]

    def __init__(self, paths, **kwargs):
        super().__init__(paths=paths, bands=kwargs['bands'])
        
    def __getitem__(self, bounds):
        batch = super().__getitem__(bounds)
        return batch

    def compute_mean_std(self, band):
        """
        Compute the mean and standard deviation of a 2D numpy array while excluding values 
        that are above the 95th percentile and below the 5th percentile.

        Parameters
        ----------
        band : string
            name of the band whose mean and std is to be calculated. must be a value in self.bands

        Returns
        -------
        mean : float
            The mean of the values within the 5th and 95th percentile range.
        std : float
            The standard deviation of the values within the 5th and 95th percentile range.
        """

        assert band in self.bands, f"Band '{band}' not available in dataset. Check path and object initialization"

        filename = [x for x in self.files if 'ortho_left' in x][0]
        with rasterio.open(filename) as ds:
            array = ds.read()

        # Flatten the 2D array to 1D
        flattened_array = array.flatten()
        
        # Compute the 5th and 95th percentiles
        lower_percentile = np.percentile(flattened_array, 5)
        upper_percentile = np.percentile(flattened_array, 95)
        
        # Create a mask to exclude values outside the 5th and 95th percentiles
        mask = (flattened_array >= lower_percentile) & (flattened_array <= upper_percentile)
        
        # Apply the mask to the array
        filtered_array = flattened_array[mask]
        
        # Calculate the mean and standard deviation of the masked array
        mean = np.mean(filtered_array)
        std = np.std(filtered_array)
        
        return mean, std

class CustomDataModule(GeoDataModule):
    """
    Custom GeoDataModule for training UNets for DeepDEM
    """

    generator = torch.Generator().manual_seed(0)
    
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
            # CustomInputDataset(kwargs["paths"], chip_size=kwargs['chip_size'], bands=self.bands),
            # batch_size=kwargs["batch_size"],
            # patch_size=kwargs["chip_size"],
            # num_workers=kwargs["num_workers"],
        )
        self.aug = None
        self.train_aug = kwargs['train_aug'] if 'train_aug' in kwargs else None
        self.val_aug = kwargs['val_aug'] if 'val_aug' in kwargs else None
        self.test_aug = kwargs['test_aug'] if 'test_aug' in kwargs else None
        self.kwargs = kwargs

        # self.train_aug = AugmentationSequential(
        #     K.Normalize(mean=self.mean, std=self.std),
        #     K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
        #     K.RandomVerticalFlip(p=0.5),
        #     K.RandomHorizontalFlip(p=0.5),
        #     data_keys=['image', 'mask'],
        #     extra_args={
        #         DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
        #     },
        # )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        if stage in ["fit", "validate"]:
            self.dataset = CustomInputDataset(**self.kwargs)

            xmin, xmax, ymin, ymax, tmin, tmax = self.dataset.bounds
            y_extent = ymax - ymin
            x_extent = xmax - xmin

            train_roi = BoundingBox(xmin, xmin + 0.8*x_extent, ymin, ymax, tmin, tmax)
            val_roi = BoundingBox(
                xmin + 0.8 * x_extent, xmax, ymin, ymin + 0.5 * y_extent, tmin, tmax
            )

            self.train_sampler = RandomBatchGeoSampler(
                self.dataset,
                batch_size=self.kwargs['batch_size'],
                size=self.kwargs['chip_size'],
                roi=train_roi,
            ) # type: ignore

            self.val_sampler = RandomBatchGeoSampler(
                self.dataset,
                batch_size=self.kwargs['batch_size'],
                size=self.kwargs['chip_size'],
                roi=val_roi,
            ) # type: ignore

        if stage == "test":
            xmin, xmax, ymin, ymax, tmin, tmax = self.dataset.bounds
            y_extent = ymax - ymin
            x_extent = xmax - xmin

            test_roi = BoundingBox(
                xmin + 0.8 * x_extent, xmax, ymin + 0.5 * y_extent, ymax, tmin, tmax
            )
            self.test_sampler = RandomBatchGeoSampler(
                self.dataset,
                batch_size=self.kwargs['batch_size'],
                size=self.kwargs['chip_size'],
                roi=test_roi,
            ) # type: ignore

    def train_dataloader(self):
        """Implement PyTorch DataLoaders for training.

        Returns:
            A dataloaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'train_dataset'/'train_sampler'
        """
        if (self.dataset is not None) and (self.train_sampler is not None):
            return DataLoader(
                dataset=self.dataset,
                batch_sampler=self.train_sampler,  # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
                pin_memory=self.kwargs["cuda"],
                shuffle=False,
            )

        msg = f"{self.__class__.__name__}.setup must define \
            a 'dataset' and a 'train_sampler'"
        raise MisconfigurationException(msg)

    def val_dataloader(self):
        """Implement PyTorch DataLoaders for validation.

        Returns:
            A dataloader specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'val_dataset'/'val_sampler'
        """
        if (self.dataset is not None) and (self.val_sampler is not None):
            return DataLoader(
                dataset=self.dataset,  # type: ignore
                batch_sampler=self.val_sampler,  # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
                pin_memory=self.kwargs["cuda"],
                shuffle=False,
            )

        msg = f"{self.__class__.__name__}.setup must define a 'dataset' and a 'val_sampler'"
        raise MisconfigurationException(msg)

    def test_dataloader(self):
        """Implement PyTorch DataLoaders for testing.

        Returns:
            A dataloader specifying test samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'/'test_sampler'.
        """
        if (self.dataset is not None) and (self.test_sampler is not None):
            return DataLoader(
                dataset=self.dataset,  # type: ignore
                batch_sampler=self.test_sampler,  # type: ignore
                num_workers=self.kwargs["num_workers"],
                collate_fn=self.kwargs["collate_fn"],
                pin_memory=self.kwargs["cuda"],
                shuffle=False,
            )

        msg = f"{self.__class__.__name__}.setup must define a 'dataset' and a 'test_sampler'"
        raise MisconfigurationException(msg)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training and self.train_aug:
            batch['image'] = self.train_aug(batch['image'])

        # print(self.trainer.training, self.trainer.validating)

        return batch

    # def __init__(self, paths: str | Iterable[str] = "data", \
    # crs: Any | None = None, res: float | None = None, \
    # bands: Sequence[str] | None = None, transforms: Callable[[dict[str, Any]], \
    # dict[str, Any]] | None = None, cache: bool = True) -> None:
    #     super().__init__(paths, crs, res, bands, transforms, cache):
    #     pass

    # for arg in args:
    #     print(arg)

    # '''
    # method to return data for training/testing/validation

    # 1. given bounds, read image
    # 2. normalize bands as appropriate
    # 3. apply transforms as appropriate
    # 4. return image
    # '''