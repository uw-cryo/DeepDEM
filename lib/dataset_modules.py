from typing import Iterable, Sequence
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, BoundingBox, stack_samples
from torchgeo.samplers import RandomBatchGeoSampler

import kornia as K
from kornia.enhance import normalize
from lightning.pytorch import LightningDataModule

import random
import rasterio
from rasterio.windows import from_bounds
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
    filename_glob = 'final_asp_dsm.tif'
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

        bands = kwargs['bands']

        if stage == 'train':
            assert (lidar_dsm_index == -1) or (lidar_dtm_index == -1), "Only one of DTM or DSM can be provided in input bands!"
            lidar_data_index = max(lidar_dsm_index, lidar_dtm_index)
        
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

        assert band in self.bands, f"Band {band} not available in dataset. Check path and object initialization"

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


class RandomPatchDataset(Dataset):
    def __init__(self, bands, datapath, chipsize, roi, len=10000, transform=True):
        self.bands = bands
        self.datapath = datapath
        self.chipsize = chipsize
        self.transform = transform
        self.len = len
        self.roi = roi 
        
        # Assuming all the bands have the same transform
        with rasterio.open(f'{datapath}/final_{bands[0]}.tif') as ds:
            self.transform = ds.transform

    def __getitem__(self, index):
        
        i = np.random.randint(self.roi[0]+1, self.roi[1] - (self.transform[0]*self.chipsize)-1, 1)[0]
        j = np.random.randint(self.roi[2]+1, self.roi[3] - (abs(self.transform[4])*self.chipsize)-1, 1)[0]
        window = from_bounds(i, j, i+(self.transform[0]*self.chipsize), 
                             j+(abs(self.transform[4])*self.chipsize), self.transform)
        
        images = []
        for band in self.bands:
            with rasterio.open(f'{self.datapath}/final_{band}.tif') as src:
                images.append(src.read(1, window=window))
        
        image = torch.tensor(np.array(images), dtype=torch.float)
        
        if self.transform:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            
            if hflip:
                image = torch.flip(image, [2])
                
            if vflip:
                image = torch.flip(image, [1])
            
        return image

    def __len__(self):
        return self.len


class CustomDataModule(LightningDataModule):
    """
    Custom GeoDataModule for training UNets for DeepDEM
    """

    generator = torch.Generator().manual_seed(0)
    
    def __init__(self, **kwargs):
        super().__init__()
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
            with rasterio.open(f'{self.kwargs['paths']}/final_{self.kwargs['bands'][0]}.tif') as ds:
                roi = ds.bounds
                
            xmin, ymin, xmax, ymax = list(roi)
            
            y_extent = ymax - ymin
            x_extent = xmax - xmin
            
            # we split the input raster vertically into train and validate regions
            x_split = xmin + self.train_split * x_extent 

            train_roi = [xmin, x_split, ymin, ymax]
            val_roi = [x_split, xmax, ymin, ymax]
            
            self.train_dataset = RandomPatchDataset(
                self.kwargs['bands'],
                self.kwargs['paths'],
                chipsize=self.kwargs['chip_size'],
                roi=train_roi,
                len=48000,
            ) # type: ignore

            self.val_dataset = RandomPatchDataset(
                self.kwargs['bands'],
                self.kwargs['paths'],
                chipsize=self.kwargs['chip_size'],
                roi=val_roi,
                len=12000,
            ) # type: ignore

        # We do not currently have a test_step, but write out a test_sampler for future use
        if stage == "test":
            xmin, ymin, xmax, ymax = list(roi)
            y_extent = ymax - ymin
            x_extent = xmax - xmin

            test_roi = [
                xmin + 0.8 * x_extent, xmax, ymin + 0.5 * y_extent, ymax
            ]
            self.test_dataset = RandomPatchDataset(
                self.kwargs['bands'],
                self.kwargs['paths'],
                chipsize=self.kwargs['chip_size'],
                roi=test_roi,
                len=6000,
            ) # type: ignore

    def train_dataloader(self):
        """Implement PyTorch DataLoaders for training.

        Returns:
            A dataloaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'train_dataset'/'train_sampler'
        """
        if (self.train_dataset is not None):
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.kwargs['batch_size'],
                num_workers=self.kwargs["num_workers"],
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
        if (self.val_dataset is not None):
            return DataLoader(
                dataset=self.val_dataset,  # type: ignore
                batch_size=self.kwargs['batch_size'],
                num_workers=self.kwargs["num_workers"],
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
        if (self.test_dataset is not None):
            return DataLoader(
                dataset=self.test_dataset,  # type: ignore
                batch_size=self.kwargs['batch_size'],
                num_workers=self.kwargs["num_workers"],
                pin_memory=self.kwargs["cuda"],
                shuffle=False,
            )

        msg = f"{self.__class__.__name__}.setup must define a 'dataset' and a 'test_sampler'"
        raise MisconfigurationException(msg)
