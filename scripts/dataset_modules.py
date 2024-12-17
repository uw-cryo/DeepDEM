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
    6. Ground truth lidar data (DSM or DTM) to train the model
    """

    is_image = True
    separate_files = True
    filename_glob = 'final_*.tif'
    filename_regex = r"final_(?P<band>[a-z_]+)\.tif"
    all_bands = [
        "asp_dsm",
        "ortho_channel1",
        "ortho_channel2",
        "ndvi",
        "nodata_mask",
        "triangulation_error",
        "lidar_data",
        "lidar_dtm_data",
    ]

    def __init__(self, paths, stage='train', **kwargs):
        lidar_dsm_index, lidar_dtm_index = -1, -1
        try:
            lidar_dsm_index = kwargs['bands'].index("lidar_data")
        except:
            pass

        try:
            lidar_dtm_index = kwargs['bands'].index("lidar_dtm_data")
        except:
            pass

        assert (lidar_dsm_index == -1) or (lidar_dtm_index == -1), "Only one of DTM or DSM can be provided in input bands!"
        lidar_data_index = max(lidar_dsm_index, lidar_dtm_index)

        bands = kwargs['bands']
        if stage == 'train':
            assert lidar_data_index != -1, "Either DTM or DSM must be specified as part of inputs!"

            # Training labels from LIDAR must be the last channel
            lidar_data = bands.pop(lidar_data_index)
            bands.append(lidar_data)

        super().__init__(paths=paths, bands=bands)
        
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

        filename = [x for x in self.files if band in x]
        assert len(filename) == 1, f"Multiple files were found for this band when querying dataset. Please examine files at {self.paths.resolve()}"
        filename = filename[0
                            ]
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
        mean = float(np.mean(filtered_array))
        std = float(np.std(filtered_array))
        
        return mean, std

class CustomDataModule(GeoDataModule):
    """
    Custom GeoDataModule for training UNets for DeepDEM
    """

    generator = torch.Generator().manual_seed(0)
    
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )
        self.aug = None
        self.train_aug = kwargs['train_aug'] if 'train_aug' in kwargs else None
        self.val_aug = kwargs['val_aug'] if 'val_aug' in kwargs else None
        self.test_aug = kwargs['test_aug'] if 'test_aug' in kwargs else None

        if 'train_split' in kwargs:
            print(f"CustomDataModule: Using the left {100*kwargs['train_split']}% of the input image for training, remaining for validation")
        self.train_split = kwargs['train_split'] if 'train_split' in kwargs else 0.8
        self.kwargs = kwargs

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
            
            # we split the input raster vertically into train and validate regions
            x_split = xmin + self.train_split * x_extent 

            train_roi = BoundingBox(xmin, x_split, ymin, ymax, tmin, tmax)
            val_roi = BoundingBox(x_split, xmax, ymin, ymax, tmin, tmax)
            
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

        # We do not currently have a test_step, but write out a test_sampler for future use
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

        return batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch