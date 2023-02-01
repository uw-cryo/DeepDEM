"""Load pixel-aligned raster stack into torchgeo.

Inspired by torchgeo implementation of 2022 IEEE GRSS Data Fusion Contest (DFC2022) dataset.
"""

import glob

import os
import sys
from typing import Callable, Dict, List, Optional
import pprint

import numpy as np
import rasterio
import torch
from torch import Tensor

import matplotlib.pyplot as plt

from rtree.index import Index, Property
from torchgeo.datasets.geo import GeoDataset
from rasterio.crs import CRS
from torchgeo.datasets.utils import BoundingBox
import rasterio.fill


class TGDSMOrthoDataset(GeoDataset):
    """Initial DSM + ortho dataset.

    Pixel-aligned rasters loaded from 1000x1000 tiles

    Dataset features:
    * Left orthoimage (panchromatic, 1.0 m per pixel spatial resolution)
    * Right orthoimage (panchromatic, 1.0 m per pixel spatial resolution)
    * Initial stereo DSM
    * Lidar DSM (ground truth)

    In the future:
    * Triangulation Error
    * Additional stereo DSM
    * Land cover mask
    * RGB?
    * ...

    Dataset format:

    * orthoimages are single-channel geotiffs
    * DEMS are single-channel geotiffs
    * (soon) masks are single-channel geotiffs with the pixel values represent the class
    """

    # TODO hardcoding stereo pair for first Easton test
    ortho_left_root = "1020010042D39D00.r100_ortho_1.0m_ba.tif"
    ortho_right_root = "1020010043455300.r100_ortho_1.0m_ba.tif"
    initial_dem_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM_holes_filled.tif"
    initial_dem_unfilled_root = (
        "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM.tif"
    )
    target_root = "USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif"
    triangulation_error_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-IntersectionErr.tif"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        input_layers: list = ["dsm", "ortho_left", "ortho_right"],
        PATCH_SIZE=256,
    ) -> None:
        """Initialize a new TGDSMOrthoDataset dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if ``split`` is invalid
        """
        super().__init__(transforms=transforms)
        self.root = root
        self.split = split
        self.transforms = transforms

        self.input_layers = input_layers

        # Should pass these in as args...
        self._crs = CRS.from_epsg(32610)
        self.res = 1.0

        self.PATCH_SIZE = PATCH_SIZE

        # self._verify() # TODO bring some sort of quality check back...

        self.index = Index(
            interleaved=False, properties=Property(dimension=3)
        )  # TODO return to 3 dimensions to search across different capture times??? Not needed right now
        # self.index.set_dimension(-2) # Was not working above???

        self.files = self._load_files()

    def __getitem__(self, query: BoundingBox):  # -> Dict[str, Any]:
        """Retrieve rasters/masks and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of rasters/masks and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        files = [hit.object for hit in hits]

        files_str = pprint.pformat(files)
        assert (
            len(files) == 1
        ), f"assuming we have non-overlapping tiles... have {len(files)}, query was {query}, files={files_str}"
        files = files[0]

        minx, maxx, miny, maxy = query[:4]

        # TODO using "dem" because could be targeting DSM or DTM & receive multiple DEM inputs...
        # TODO may have a rare bug appear when rasters are not all 256x256???
        # Update 6/2022 not sure what that bug was
        # TODO 7/2022 it is still there
        # ortho_left_fn = self._load_image(files["ortho_left"])
        # UPDATE This was orthoimage bounds related... off by 1 when creating the rasters.
        # Problem should be eliminated in the move away from 1km^2 tile-based datasets.

        # TODO find the union of the nodata masks for each input & pass this info to NN

        with rasterio.open(files["ortho_left"]) as ortho_left_full:
            ortho_left_unfilled = ortho_left_full.read(
                1,
                masked=True,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, ortho_left_full.transform
                ),
            )

            # Try rasterio fill for the small gaps in orthoimages
            ortho_left = rasterio.fill.fillnodata(ortho_left_unfilled)

        with rasterio.open(files["ortho_right"]) as ortho_right_full:
            ortho_right_unfilled = ortho_right_full.read(
                1,
                masked=True,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, ortho_right_full.transform
                ),
            )

            # Try rasterio fill for the small gaps in orthoimages
            ortho_right = rasterio.fill.fillnodata(ortho_right_unfilled)

        with rasterio.open(files["initial_dem"]) as initial_dem_full:
            initial_dem = initial_dem_full.read(
                1,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, initial_dem_full.transform
                ),
                masked=False,  # initial DEM cannot have nodata holes
            )

            # TODO try removing temporal offset from initial dem snow /ice melt in training

        with rasterio.open(files["initial_dem_unfilled"]) as initial_dem_unfilled_full:
            initial_dem_unfilled = initial_dem_unfilled_full.read(
                1,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, initial_dem_unfilled_full.transform
                ),
                masked=True,
            )
            # Read in the nodata mask for the initial DEM (i.e. the raster before it was inpainted)
            nodata_mask = initial_dem_unfilled_full.read_masks(
                1,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, initial_dem_unfilled_full.transform
                ),
            )

        with rasterio.open(files["triangulation_error"]) as tri_error_full:
            tri_error = tri_error_full.read(
                1,
                masked=True,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, tri_error_full.transform
                ),
            )
            tri_error = rasterio.fill.fillnodata(tri_error)

        sample = {"crs": self.crs, "bbox": query}

        assert initial_dem_unfilled.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: initial_dem_unfilled = f{initial_dem_unfilled.shape}"
        assert initial_dem.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: initial_dem = f{initial_dem.shape}"
        assert ortho_left.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: ortho_left = f{ortho_left.shape}, {files['ortho_left']}"
        assert ortho_right.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: ortho_right = f{ortho_right.shape}"
        assert tri_error.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: tri_error = f{tri_error.shape}"

        assert nodata_mask.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: dem_mask = f{nodata_mask.shape}"

        if self.split == "train":
            # Retrieve Ground Truth e.g. lidar DEM
            with rasterio.open(files["target"]) as target_full:
                target = target_full.read(
                    1,
                    window=rasterio.windows.from_bounds(
                        minx, miny, maxx, maxy, target_full.transform
                    ),
                    masked=True,  # lidar DEM should not have holes
                )

            assert target.shape == (
                self.PATCH_SIZE,
                self.PATCH_SIZE,
            ), f"Wrong sized patch for query {query}: target = f{target.shape}"

            sample["target"] = torch.from_numpy(target)

        # TODO make checking & stacking work properly for different layer combinations
        # i.e. a dict of input layer -> array and then stack with a list comprehension

        # TODO nodata_mask could be the union of initial DEM mask (NOT the _hole_filled.tif), ortho masks
        if "nodata_mask" in self.input_layers:
            sample["inputs"] = torch.from_numpy(
                np.stack(
                    [initial_dem, ortho_left, ortho_right, tri_error, nodata_mask],
                    axis=0,
                )
            )
        else:
            sample["inputs"] = torch.from_numpy(
                np.stack([initial_dem, ortho_left, ortho_right], axis=0)
            )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:

        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image/dem/mask
        """
        directories = self.root
        if not isinstance(directories, list):
            directories = [directories]
        files = []
        for directory in directories:
            # TODO need to differentiate between left & right orthoimages???
            pattern = os.path.join(
                directory, "**", "files_to_zip", self.initial_dem_root
            )
            print("Initial DEM pattern:", pattern)
            initial_dems = glob.glob(pattern, recursive=True)

            print("Adding files to list & spatial index...")
            initial_dems_str = "\n".join(initial_dems)
            print(f"Have {len(initial_dems)} tiles")

            for i, initial_dem in enumerate(sorted(initial_dems)):

                initial_dem_unfilled = initial_dem.replace(
                    self.initial_dem_root, self.initial_dem_unfilled_root
                )
                ortho_left = initial_dem.replace(
                    self.initial_dem_root, self.ortho_left_root
                )
                ortho_right = initial_dem.replace(
                    self.initial_dem_root, self.ortho_right_root
                )

                triangulation_error = initial_dem.replace(
                    self.initial_dem_root, self.triangulation_error_root
                )

                tile_dict = dict(
                    ortho_left=ortho_left,
                    ortho_right=ortho_right,
                    initial_dem=initial_dem,
                    initial_dem_unfilled=initial_dem_unfilled,
                    triangulation_error=triangulation_error,
                )

                if self.split == "train":
                    target = initial_dem.replace(
                        self.initial_dem_root, self.target_root
                    )
                    # print("Target glob pattern:", target)
                    target_paths = glob.glob(target)
                    assert (
                        len(target_paths) == 1
                    ), f"Should have exactly 1 ground truth DEM, got {target_paths}"
                    target = target_paths[0]
                    tile_dict["target"] = target

                files.append(tile_dict)

                with rasterio.open(ortho_left) as f:
                    minx, miny, maxx, maxy = f.bounds
                    # Used in chesapeake.py, but we don't care about time yet
                    mint: float = 0
                    maxt: float = sys.maxsize
                    coords = (minx, maxx, miny, maxy, mint, maxt)

                # Add files to spatial index
                self.index.insert(i, coords, tile_dict)

        print(f"Created dataset with {len(files)} tiles")
        return files

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        raise NotImplementedError
