"""Define torchgeo RasterDatasets that will be used to load our data"""
from torchgeo.datasets import RasterDataset

class InputRasterData(RasterDataset):
    """
    Torchgeo wrapper for training data
    """
    is_image=True
    separate_files=True
    # filename_glob = "*DEM*"
    all_bands = ["final_stereo_DEM_filled", "final_ortho_left", "final_ortho_right", 
                 "final_stereo_triangulation_errors", "final_nodata_mask"]

class LabelRasterData(RasterDataset):
    """
    Torchgeo wrapper for training labels
    """
    is_image=True
    separate_files=True
    filename_glob = "*lidar*"
    all_bands = ["DEM"]
