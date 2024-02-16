# misc imports
from pathlib import Path
# from typing import Dict
import numpy as np

# The DeepDEM framework uses a UNet for DEM refinement
# from UNet import UNet

# pytorch imports
# from torch import optim, nn, Tensor
# from torch.utils.data import DataLoader
# import torch

# pytorch-lightning imports
# from lightning.pytorch.utilities.types import OptimizerLRScheduler
import lightning as L

# torchgeo imports
from torchgeo.datasets import RasterDataset, stack_samples, BoundingBox
from torchgeo.samplers import GridGeoSampler

from modules.modules import CustomGeoDataModule, DeepDEMModel
from modules.dataset_modules import InputRasterData, LabelRasterData

if __name__ == '__main__':

    # specify paths of input dataset and the labels
    input_dataset = InputRasterData('/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/baker_csm_stack/cropped_rasters/input_channels')
    label_dataset = LabelRasterData('/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/baker_csm_stack/cropped_rasters/target_labels')

    # RasterDatasets support '&' operations for merging
    mtbaker_dataset = input_dataset & label_dataset

    xmin, xmax, ymin, ymax, tmin, tmax = mtbaker_dataset.index.bounds
    crs = mtbaker_dataset.crs
    xsplit = np.floor(0.8*(xmax-xmin))+xmin

    # Sampler
    train_sampler = GridGeoSampler(mtbaker_dataset, size=256, stride=248, roi=BoundingBox(xmin, xsplit, ymin, ymax, tmin, tmax))
    test_sampler = GridGeoSampler(mtbaker_dataset, size=256, stride=248, roi=BoundingBox(xsplit, xmax, ymin, ymax, tmin, tmax))

    datamodule_params = {
        'dataset_class': RasterDataset,
        'batch_size': 16, 
        'patch_size': 256,
        'num_workers':8,
        'dataset':mtbaker_dataset,
        
        'train_dataset':mtbaker_dataset,
        'val_dataset':mtbaker_dataset,
        'test_dataset':mtbaker_dataset,
        
        'train_sampler':train_sampler,
        'test_sampler':test_sampler,

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
