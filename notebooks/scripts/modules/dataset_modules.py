"""Define torchgeo RasterDatasets that will be used to load our data"""
from torchgeo.datasets import RasterDataset

class InputRasterData(RasterDataset):
    """
    Torchgeo wrapper for training data
    """
    is_image=True
    separate_files=True
    filename_glob = "final*DEM*.tif"
    all_bands = ["stereo_DEM_filled", "ortho_left", "ortho_right", 
                 "triangulation_errors", "nodata_mask"]

class LabelRasterData(RasterDataset):
    """
    Torchgeo wrapper for training labels
    """
    is_image=False
    separate_files=True
    filename_glob = "final*lidar*.tif"
    all_bands = ["DEM"]
