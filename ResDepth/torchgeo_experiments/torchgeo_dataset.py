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

    
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        input_layers: list = ["dsm", "ortho_left", "ortho_right"],
        PATCH_SIZE=256,
        crs="EPSG:32610",
        res=1.0,
        initial_dem_root=None,
        dataset="baker2015_singletile"
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

        # dataset = "baker2015_singletile"
        if dataset == "baker2015":
            # TODO hardcoding stereo pair for first Easton test
            ortho_left_root = "1020010042D39D00.r100_ortho_1.0m_ba.tif"
            ortho_right_root = "1020010043455300.r100_ortho_1.0m_ba.tif"
            initial_dem_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM_holes_filled.tif"
            initial_dem_unfilled_root = (
                "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM.tif"
            )
            target_root = "USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif"
            triangulation_error_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-IntersectionErr.tif"
        elif dataset == "baker2015_melt_adjusted":
            # TODO hardcoding stereo pair for first Easton test
            ortho_left_root = "1020010042D39D00.r100_ortho_1.0m_ba.tif"
            ortho_right_root = "1020010043455300.r100_ortho_1.0m_ba.tif"
            initial_dem_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM_holes_filled_snow_median_subtracted.tif"
            initial_dem_unfilled_root = (
                "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM.tif"
            )
            target_root = "USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif"
            triangulation_error_root = "WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-IntersectionErr.tif"
        elif dataset == "baker2015_singletile":
            # TODO properly fill in holes
            initial_dem_unfilled_root = "try_pc_align_to_lidar_15m_maxdisp_rotationallowed-1.0m-DEM.tif"
            initial_dem_root = "try_pc_align_to_lidar_15m_maxdisp_rotationallowed-1.0m-DEM_holes_filled.tif"
            # target_root = "try_pc_align_to_lidar_15m_maxdisp_rotationallowed-1.0m-DEM_holes_filled.tif"
            ortho_left_root = "final_ortho_left_1.0m.tif"
            ortho_right_root = "final_ortho_right_1.0m.tif"
            triangulation_error_root = "try_pc_align_to_lidar_15m_maxdisp_rotationallowed-1.0m-IntersectionErr_holes_filled.tif"
            target_root = "mosaic_full128_USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_holes_filled.tif"
        elif dataset == "scg2019_csm":
            initial_dem_unfilled_root = "aligned_stereo_1.0m-DEM.tif"
            initial_dem_root = "aligned_stereo_1.0m-DEMgdal_fillnodata_md500_si3.tif"
            ortho_left_root = "final_ortho_left_1.0m.tif"
            ortho_right_root = "final_ortho_right_1.0m.tif"
            triangulation_error_root = "aligned_stereo_1.0m-IntersectionErrgdal_fillnodata_md500_si3.tif"
            target_root = "scg_merged_lidar_dsm_1.0m-DEM_interpolate_na.tif" 

        elif dataset == "scg2019":
            # Try with SCG
            # Recheck ordering but changing anyway to match the 
            ortho_left_root = "WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001_ortho_1.0m_interpolate_na.tif" #"WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001_ortho_1.0m_ba_aligned_with_pc_align_and_initial_ba.tif"
            ortho_right_root = "WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001_ortho_1.0m_interpolate_na.tif" #"WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001_ortho_1.0m_ba_aligned_with_pc_align_and_initial_ba.tif""WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001_ortho_1.0m_ba_aligned_with_pc_align_and_initial_ba.tif"
            initial_dem_root = "scg_aligned_asp_dsm_1.0m-DEM_holes_filled.tif"
            initial_dem_unfilled_root = "scg_aligned_asp_dsm_1.0m-DEM.tif"#_interpolate_na.tif"
            # target_unfilled_root = "scg_merged_lidar_dsm_1.0m-DEM.tif"  # TODO use lidar sparsity
            target_root = "scg_merged_lidar_dsm_1.0m-DEM_interpolate_na.tif"
            # TODO get the target lidar nodata mask too
            # triangulation_error_root = "scg_aligned_asp_dsm_1.0m-IntersectionErr.tif"
            # triangulation_error_root = "scg_aligned_asp_dsm_1.0m-IntersectionErr.tif"
            triangulation_error_root = "scg_aligned_asp_dsm_1.0m-IntersectionErr_interpolate_na.tif"
        self.dataset = dataset
        self.ortho_left_root = ortho_left_root
        self.ortho_right_root = ortho_right_root
        self.initial_dem_root = initial_dem_root
        self.initial_dem_unfilled_root = initial_dem_unfilled_root
        self.target_root = target_root
        self.triangulation_error_root = triangulation_error_root
        

        # TODO hacky workaround to use melt-corrected raster instead as the filepath. File paths have to be easily configured
        if initial_dem_root is not None:
            self.initial_dem_root = initial_dem_root

        super().__init__(transforms=transforms)
        self.root = root
        self.split = split
        self.transforms = transforms

        self.input_layers = input_layers

        # Should pass these in as args...
        self._crs = CRS.from_string(crs)
        self.res = res
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
                masked=True,  # try allowing nodata in initial DEM
                # masked=False,  # initial DEM cannot have nodata holes
            )

            # TODO try removing temporal offset from initial dem snow /ice melt in training

            # TODO some initial DEMs have large holes remaining - keep this?
            initial_dem = rasterio.fill.fillnodata(initial_dem)
        
        # print("Read DEM")

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

        if self.split == "train" or self.split == "val":
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

            if self.dataset == "baker2015" or self.dataset == "baker2015_melt_adjusted":
                pattern = os.path.join(
                    directory, "**", "files_to_zip", self.initial_dem_root
                )
            else:
                pattern = os.path.join(
                    directory, self.initial_dem_root  # TODO make this work if there is a single file
                )

            patterns = [pattern]
            # patterns = [pattern_single, pattern_tiles]
            initial_dems = []
            for pattern in patterns:
                print("Initial DEM pattern:", pattern)
                initial_dems.extend(glob.glob(pattern, recursive=True))

            print("Adding files to list & spatial index...")
            # initial_dems_str = "\n".join(initial_dems)
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

                if self.split == "train" or self.split == "val":
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
                
                # Used in chesapeake.py, but we don't care about time yet
                mint: float = 0
                maxt: float = sys.maxsize
                
                with rasterio.open(initial_dem) as demf:
                    dminx, dminy, dmaxx, dmaxy = demf.bounds
                    coords = (
                        max(minx, dminx),
                        min(maxx, dmaxx),
                        max(miny, dminy),
                        min(maxy, dmaxy),
                        mint,
                        maxt
                    )

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


if __name__ == "__main__":
    # Quick test of torchgeo UnionDataset functionality
    # geo.py was fixed in https://github.com/microsoft/torchgeo/pull/786
    TILE_SIZE = 512

    input_layers = [
        "dsm",
        "ortho_left",
        "ortho_right",
        "triangulation_error",
        "nodata_mask",
    ]

    dataset_SCG = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK", split="train", dataset="scg2019", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1)
    dataset_Baker2015 = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/dataset_processing/baker_csm/baker_csm_stack", split="train", dataset="baker2015_singletile", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1)

    merged_dataset = dataset_SCG | dataset_Baker2015
    dataset_SCG2 = TGDSMOrthoDataset("/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/SCG_ALIGNED_STACK", split="train", dataset="scg2019", input_layers=input_layers, PATCH_SIZE=TILE_SIZE, crs="EPSG:32610", res=1)

    # Demonstrate the power of merging two datasets (from different regions but same CRS)
    merged_dataset = dataset_SCG & dataset_SCG2

    print(len(merged_dataset))
    minx = 640000
    miny = 5355400
    maxx = minx+TILE_SIZE
    maxy = miny + TILE_SIZE
    mint = 0
    maxt = 9.223372036854776e+18
    out = merged_dataset.__getitem__(BoundingBox(minx, maxx, miny, maxy, mint, maxt))
    print(out["inputs"].shape)
