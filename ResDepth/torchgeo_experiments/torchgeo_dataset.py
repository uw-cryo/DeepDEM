"""Load pixel-aligned raster stack into torchgeo.

Inspired by torchgeo implementation of 2022 IEEE GRSS Data Fusion Contest (DFC2022) dataset.
"""

import glob

import os
import sys
from typing import Callable, Dict, List, Optional

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
    initial_dem_root = "pc_align_tr-trans_source_1.0m-DEM_holes_filled.tif"
    target_root = "USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif"

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
        assert split == "train"  # TODO make "val" "test" usable
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
        # print(self.index.get_bounds())
        # print(query)
        # print("FIRST, trying to query with known coordinates that should be in the dataset")
        # query_test = (582000, 582256, 5300000, 5300256, 0.0, 9.2233e18)
        # hits_test = self.index.intersection(tuple(query_test), objects=True)
        # print(list(hits_test))

        # print("\n -------\n")
        # print(tuple(query))
        # print(self.index)
        hits = self.index.intersection(tuple(query), objects=True)
        # print(hits)
        files = [hit.object for hit in hits]
        assert (
            len(files) == 1
        ), f"assuming we have non-overlapping tiles... have {len(files)}, query was {query}"
        files = files[0]
        # print(query)
        minx, maxx, miny, maxy = query[:4]
        # print("__getitem__: ", minx, maxx, miny, maxy)
        # query_box = shapely.geometry.box(*query[:4]).envelope
        # print(query_box)
        # query_geom = shapely.geometry.mapping(query_box)
        # print(query_geom)

        # ortho_left, _ = rasterio.mask.mask(ortho_left_full, [(minx, miny, maxx, maxy)], crop=True, all_touched=True)

        # TODO using "dem" because could be targeting DSM or DTM & receive multiple DEM inputs...
        # TODO may have a rare bug appear when rasters are not all 256x256???
        # Update 6/2022 not sure what that bug was
        # TODO 7/2022 it is still there
        # ortho_left_fn = self._load_image(files["ortho_left"])
        # UPDATE This was orthoimage bounds related... off by 1 when creating tiffs...

        # TODO find the union of the nodata masks for each input & pass this info to NN

        with rasterio.open(files["ortho_left"]) as ortho_left_full:
            ortho_left = ortho_left_full.read(
                1,
                masked=True,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, ortho_left_full.transform
                ),
            )

            # Try rasterio fill for the small gaps in orthoimages
            ortho_left = rasterio.fill.fillnodata(ortho_left)

        with rasterio.open(files["ortho_right"]) as ortho_right_full:
            ortho_right = ortho_right_full.read(
                1,
                masked=True,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, ortho_right_full.transform
                ),
            )

            # Try rasterio fill for the small gaps in orthoimages
            ortho_right = rasterio.fill.fillnodata(ortho_right)

        with rasterio.open(files["initial_dem"]) as initial_dem_full:
            initial_dem = initial_dem_full.read(
                1,
                window=rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, initial_dem_full.transform
                ),
            )

        # Goal of mask: inform the network of pixels that are missing data in at least one raster.
        nodata_mask = np.logical_and(initial_dem, ortho_left, ortho_right)

        if self.split == "train":
            # Retrieve Ground Truth e.g. lidar DEM
            with rasterio.open(files["target"]) as target_full:
                target = target_full.read(
                    1,
                    window=rasterio.windows.from_bounds(
                        minx, miny, maxx, maxy, target_full.transform
                    ),
                )

        sample = {"crs": self.crs, "bbox": query}

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
        assert nodata_mask.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: dem_mask = f{nodata_mask.shape}"

        assert target.shape == (
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        ), f"Wrong sized patch for query {query}: target = f{target.shape}"

        # TODO make this behave properly for different layer combinations

        # TODO nodata_mask could be the union of initial DEM mask (NOT the _hole_filled.tif), ortho masks
        if "nodata_mask" in self.input_layers:
            sample["inputs"] = torch.from_numpy(
                np.stack([initial_dem, ortho_left, ortho_right, nodata_mask], axis=0)
            )
        else:
            sample["inputs"] = torch.from_numpy(
                np.stack([initial_dem, ortho_left, ortho_right], axis=0)
            )

        sample["target"] = torch.from_numpy(target)

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
        directory = self.root
        # TODO need to differentiate between left & right orthoimages???
        pattern = os.path.join(
            directory, "**", "files_to_zip", self.initial_dem_root
        )  # + ".tif")
        print("Initial DEM pattern:", pattern)
        initial_dems = glob.glob(pattern, recursive=True)

        files = []
        print("Adding files to list & spatial index...")
        initial_dems_str = "\n".join(initial_dems)
        print(f"Initial DEMs:\n{initial_dems_str}")
        print("\n\n")
        print(f"Have {len(initial_dems)} tiles")
        input("Hit enter to continue:")
        for i, initial_dem in enumerate(sorted(initial_dems)):
            print(f"Initial DEM {initial_dem}")

            ortho_left = initial_dem.replace(
                self.initial_dem_root, self.ortho_left_root
            )
            ortho_right = initial_dem.replace(
                self.initial_dem_root, self.ortho_right_root
            )

            if self.split == "train":
                target = initial_dem.replace(self.initial_dem_root, self.target_root)
                print("Target glob pattern:", target)
                target_paths = glob.glob(target)
                assert (
                    len(target_paths) == 1
                ), f"Should have exactly 1 ground truth DEM, got {target_paths}"
                target = target_paths[0]

                tile_dict = dict(
                    ortho_left=ortho_left,
                    ortho_right=ortho_right,
                    initial_dem=initial_dem,
                    target=target,
                )
                files.append(tile_dict)

                with rasterio.open(ortho_left) as f:
                    minx, miny, maxx, maxy = f.bounds
                    # Used in chesapeake.py, but we don't care about time
                    mint: float = 0
                    maxt: float = sys.maxsize
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                # Add files to spatial index
                self.index.insert(i, coords, tile_dict)

                # print(f"Added {i} for {coords} with tile_dict:")
                # pprint.pprint(tile_dict, indent=2)
                # print("\n")
            else:
                raise NotImplementedError  # test mode not implemented yet

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
