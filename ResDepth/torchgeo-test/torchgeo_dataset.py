# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# TODO modified DFC2022 dataset as it appears to be closest to DSMs+ortho in terms of multiple pixel-aligned rasters
# Original
# """2022 IEEE GRSS Data Fusion Contest (DFC2022) dataset."""

import glob
# from msilib.schema import Directory
import os
from re import S
import sys
from typing import Callable, Dict, List, Optional, Sequence
from unittest.mock import call

import shapely

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib import colors
from rasterio.enums import Resampling
from torch import Tensor

# Just to import ResDepth model for quick test
import sys
sys.path.insert(0, "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/ResDepth/")
from lib.UNet import UNet

import numpy as np
import torchvision


from torchgeo.datasets.geo import VisionDataset
from torchgeo.datasets.utils import check_integrity, extract_archive, percentile_normalization

import pprint
import matplotlib.pyplot as plt

from rtree.index import Index, Property
from torchgeo.samplers import RandomBatchGeoSampler # appropriate for tile based dataset
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets import stack_samples
from rasterio.crs import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import BoundingBox
import rasterio

import random

# TODO Set consistent seeds (want to see variety of inputs right now)
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


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

    * Data collected from 19 regions in France

    Dataset format:

    * orthoimages are single-channel geotiffs
    * DEMS are single-channel geotiffs
    * (soon) masks are single-channel geotiffs with the pixel values represent the class

    Dataset classes: leaving this for later

    0. n/a
    1. ...

    If you use this dataset in your research, please cite the following paper:
    * ...

    """  # noqa: E501

    # image_root = "BDORTHO"
    # dem_root = "RGEALTI"
    # target_root = "UrbanAtlas"
    # 1020010042D39D00.r100_ortho_1.0m_ba(3).tif
    # 1020010043455300.r100_ortho_1.0m_ba(4).tif
    # 1020010042D39D00.r100_ortho_1.0m_ba(4).tif
    # pc_align_tr-trans_source_1.0m-DEM.tif
    # USGS_LPC_WA_MtBaker_2015_10UEU8597_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_hs.tif
    # USGS_LPC_WA_MtBaker_2015_10UEU8598_LAS_2017_32610_first_filt_v1.3_1.0m-DEM.tif
    # pc_align_tr-trans_source_1.0m-DEM.tif
    # 1020010043455300.r100_ortho_1.0m_ba(3).tif

    # TODO hardcoding stereo pair for first Easton test
    # image_dir = f"{run_name}/files_to_zip/"
    # Remove the .tif to make this substitution work
    ortho_left_root = "1020010042D39D00.r100_ortho_1.0m_ba"
    ortho_right_root = "1020010043455300.r100_ortho_1.0m_ba"
    initial_dem_root = "pc_align_tr-trans_source_1.0m-DEM_hole_fill"
    target_root = "USGS_LPC_WA_MtBaker_2015_*_LAS_2017_32610_first_filt_v1.3_1.0m-DEM_hole_fill"
    # lidar_dsm = "USGS_LPC_WA_MtBaker_2015_10UEU8597_LAS_2017_32610_first_filt_v1.3_1.0m-DEM.tif"




    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new TGDSMOrthoDataset dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` is invalid
        """
        super().__init__(transforms=transforms)
        # print(self.index)
        # assert split in self.metadata # TODO return to checking
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        # Should pass these in as args...
        self._crs = CRS.from_epsg(32610)
        self.res = 1.0

        # self._verify() # TODO bring some sort of quality check back...

        # self.class2idx = {c: i for i, c in enumerate(self.classes)}

        self.index = Index(interleaved=False, properties=Property(dimension=3)) # TODO return to 3 dimensions to search across different capture times??? Not needed right now
        # self.index.set_dimension(-2) # Was not working above???
        # print(self.index.get_dimension())

        self.files = self._load_files()

        # self.index.insert(i, coords, filepath)
        #                 {
        #                     "naip-new": row["properties"]["naip-new"],
        #                     "naip-old": row["properties"]["naip-old"],
        #                     "landsat-leaf-on": row["properties"]["landsat-leaf-on"],
        #                     "landsat-leaf-off": row["properties"]["landsat-leaf-off"],
        #                     "lc": row["properties"]["lc"],
        #                     "nlcd": row["properties"]["nlcd"],
        #                     "buildings": row["properties"]["buildings"],
        #                     "prior_from_cooccurrences_101_31_no_osm_no_buildings": prior_fn,  # noqa: E501
        #                 },

    def __getitem__(self, query: BoundingBox): # -> Dict[str, Any]:
        """Retrieve rasters/masks and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of rasters/masks and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        print(self.index.get_bounds())
        print(query)
        hits = self.index.intersection(tuple(query), objects=True)
        files= [hit.object for hit in hits]
        assert len(files) == 1, f"assuming we have non-overlapping tiles... have {len(files)}"
        files= files[0]
        # print(query)
        minx, maxx, miny, maxy = query[:4]
        print("__getitem__: ", minx, maxx, miny, maxy)
        # query_box = shapely.geometry.box(*query[:4]).envelope
        # print(query_box)
        # query_geom = shapely.geometry.mapping(query_box)
        # print(query_geom)

        # ortho_left, _ = rasterio.mask.mask(ortho_left_full, [(minx, miny, maxx, maxy)], crop=True, all_touched=True)

        # TODO using "dem" because could be targeting DSM or DTM & receive multiple DEM inputs...
        # TODO may have a rare bug appear when rasters are not all 256x256???
        # ortho_left_fn = self._load_image(files["ortho_left"])
        with rasterio.open(files["ortho_left"]) as ortho_left_full:
            # print(files["ortho_left"])
            # print("Tile bounds:", ortho_left_full.bounds)
            ortho_left = ortho_left_full.read(1, window=rasterio.windows.from_bounds(minx, miny, maxx, maxy, ortho_left_full.transform))

        # print("Output shape: ortho left=", ortho_left.shape)
        with rasterio.open(files["ortho_right"]) as ortho_right_full:
            ortho_right = ortho_right_full.read(1, window=rasterio.windows.from_bounds(minx, miny, maxx, maxy, ortho_right_full.transform))

        # print(ortho_right.shape)

        with rasterio.open(files["initial_dem"]) as initial_dem_full:
            initial_dem = initial_dem_full.read(1, window=rasterio.windows.from_bounds(minx, miny, maxx, maxy, initial_dem_full.transform))

        # print(initial_dem.shape)

        # inputs = torch.cat(tensors=[initial_dem, ortho_left, ortho_right], dim=0)

        # sample = {"inputs": inputs}

        if self.split == "train":
            # Retrieve Ground Truth e.g. lidar DEM
            with rasterio.open(files["target"]) as target_full:
                target = target_full.read(1, window=rasterio.windows.from_bounds(minx, miny, maxx, maxy, target_full.transform))

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample = {"crs": self.crs, "bbox": query}
        print(initial_dem.shape, ortho_left.shape, ortho_right.shape)
        sample["inputs"] = torch.from_numpy(np.stack([initial_dem, ortho_left, ortho_right], axis=0))
        sample["target"] = torch.from_numpy(target)

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
        directory = self.root # all in 1 dir for now os.path.join(self.root, self.metadata[self.split]["directory"])
        # TODO need to differentiate between left & right orthoimages???
        pattern = os.path.join(directory, "**", "files_to_zip", self.initial_dem_root + ".tif")
        print(pattern)
        initial_dems = glob.glob(
            pattern, recursive=True
        )

        files = []
        print("Adding files to list & spatial index...")
        for i, initial_dem in enumerate(sorted(initial_dems)):
            print(initial_dem)
            print(os.path.basename(initial_dem))

            ortho_left = initial_dem.replace(self.initial_dem_root, self.ortho_left_root)
            ortho_right = initial_dem.replace(self.initial_dem_root, self.ortho_right_root)

            if self.split == "train":
                target = initial_dem.replace(self.initial_dem_root, self.target_root)
                target_paths = glob.glob(target)
                assert len(target_paths) == 1
                target = target_paths[0]
                # target = f"{os.path.splitext(target)[0]}_UA2012.tif"
                tile_dict = dict(ortho_left=ortho_left, ortho_right=ortho_right, initial_dem=initial_dem, target=target)
                files.append(tile_dict)

                with rasterio.open(initial_dem) as f:
                    minx, miny, maxx, maxy = f.bounds
                    # Used in chesapeake.py, don't care about time
                    mint: float = 0
                    maxt: float = sys.maxsize
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    # TODO check that other rasters have matching bounds!

                # Add files to spatial index
                self.index.insert(i, coords, tile_dict)
            else:
                assert 0, "test mode not implemented yet"

        # pprint.pprint(files, indent=2)

        # files = []
        # for initial_dem in sorted(initial_dems):

        #     dem = image.replace(self.inital_dem_root, self.dem_root)
        #     dem = f"{os.path.splitext(dem)[0]}_RGEALTI.tif"

        #     if self.split == "train":
        #         target = image.replace(self.image_root, self.target_root)
        #         target = f"{os.path.splitext(target)[0]}_UA2012.tif"
        #         files.append(dict(ortho_left=image, ortho_right=dem, initial_dem=initial_dem, target=target))
        #     else:
        #         assert 0, "test mode not implemented"
        #         files.append(dict(image=image, dem=dem))

        return files

    # def _load_image(self, path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
    #     """Load a single image.

    #     Args:
    #         path: path to the image
    #         shape: the (h, w) to resample the image to

    #     Returns:
    #         the image
    #     """
    #     with rasterio.open(path) as f:
    #         array: "np.typing.NDArray[np.float_]" = f.read(
    #             out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
    #         )
    #         tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
    #         return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target DSM for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            ) # TODO doublecheck the Resampling should stay in
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        exists = []
        for split_info in self.metadata.values():
            exists.append(
                os.path.exists(os.path.join(self.root, split_info["directory"]))
            )

        if all(exists):
            return

        # Check if .zip files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

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
        ncols = 2
        image = sample["image"][:3]
        image = image.to(torch.uint8)  # type: ignore[attr-defined]
        image = image.permute(1, 2, 0).numpy()

        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(self.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation=None)
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation=None)
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation=None)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

def hillshade(array, azimuth=315, angle_altitude=45):
    """Borrow hillshade 
    
    https://github.com/uw-cryo/wv_stereo_processing/blob/54f3e82f313773e57dea8c3af5f05bb69e6b0a68/notebooks/gm_aso_dg_comparison.ipynb"""

    # Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2


class ResDepthLM(pl.LightningModule):
    # TODO have to make sure specification of model matches with intended input rasters / other inputs
    def __init__(self, model_args):
        """Initialize the model itself"""
        super().__init__()
        self.model = UNet(**model_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass with the given inputs"""
        return self.model(x)

    def training_step(self, batch): # TODO *args, **kwargs) -> STEP_OUTPUT:
        x, gt = batch
        out = self.forward(x)
        loss = F.l1_loss(out, gt)
        
        # return super().training_step(*args, **kwargs)
        return loss

    def validation_step(self, batch):#, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        # return super().validation_step(*args, **kwargs)
        x, gt = batch
        out = self.forward(x)
        loss = F.l1_loss(out, gt)
        

        # Log metrics
        metrics = {"val_loss": loss}
        self.log_dict(metrics)

        return metrics

        # return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
        return optimizer



if __name__ == "__main__":
    directory = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/torchgeo_dataset" 
    dataset = TGDSMOrthoDataset(directory, "train", None, False)

    # First attempt to use UnionDataset did not work. Investigate docs & implementation
    # directory1 = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/torchgeo_dataset/pc_laz_prep_full_outputs_20220314_easton_original_bounds" 
    # directory2 = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/torchgeo_dataset/pc_laz_prep_full_outputs_20220315_easton_8599" 
    # directory3 = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/torchgeo_dataset/pc_laz_prep_full_outputs_20220314_easton_8597"
    # dataset1 = TGDSMOrthoDataset(directory1, "train", None, False)
    # dataset2 = TGDSMOrthoDataset(directory2, "train", None, False)
    # dataset3 = TGDSMOrthoDataset(directory3, "train", None, False)
    # dataset = dataset1 | dataset2 | dataset3
    # TODO why does this fail AssertionError: assuming we have non-overlapping tiles... have 0
    # We get all 
    # adding dataset 0 to the new index
    #  added 0 hit.bounds=[585022.5, 585982.5, 5398013.5, 5398985.5, 0.0, 9.223372036854776e+18]
    #  added 1 hit.bounds=[585022.5, 585982.5, 5399013.5, 5399985.5, 0.0, 9.223372036854776e+18]
    # adding dataset 2 to the new index
    #  added 2 hit.bounds=[585022.5, 585982.5, 5397013.5, 5397985.5, 0.0, 9.223372036854776e+18]
    # BoundingBox(minx=585604.5, maxx=585860.5, miny=5398077.5, maxy=5398333.5, mint=0.0, maxt=9.223372036854776e+18)

    BATCH_SIZE = 2
    sampler = RandomBatchGeoSampler(dataset, batch_size=BATCH_SIZE, size=256, length=12)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=stack_samples)

    # for b,batch in enumerate(dataloader):
    #     print("Plotting batch")
    #     # print(batch)

    #     # Plot example
    #     image = batch["inputs"]
    #     target = batch["target"]

    #     # print(image.shape)
    #     # print(target.shape)
    #     fig, ax = plt.subplots(BATCH_SIZE, 4)
    #     plt.suptitle(f"Batch of size {image.shape[0]}")

    #     for i in range(BATCH_SIZE):
    #         initial_dem = image[i][0].numpy().squeeze()
    #         ax[i][0].imshow(hillshade(initial_dem), cmap="gray")
    #         ax[i][0].imshow(initial_dem, alpha=0.5)
    #         ax[i][0].set_title("initial DEM")

    #         ax[i][1].imshow(image[i][1].numpy().squeeze(), cmap="gray")
    #         ax[i][1].set_title("ortho left")

    #         ax[i][2].imshow(image[i][2].numpy().squeeze(), cmap="gray")
    #         ax[i][2].set_title("ortho right")

    #         gt_dem = target[i].numpy().squeeze()
    #         ax[i][3].imshow(hillshade(gt_dem), cmap="gray")
    #         ax[i][3].imshow(gt_dem, alpha=0.5)
    #         ax[i][3].set_title("lidar DEM")

    #     plt.tight_layout()
    #     plt.savefig(f"out_batch_{b}.png")
    #     # plt.show()


    run_folder = "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/resdepth-output/2021-09-30_14-15_ResDepth-stereo/"
    def call_model(model, model_input_tensor : torch.Tensor):
        """Helper: given pytorch model & input rasters, return model output as numpy array"""
        # model_input = np.stack(inputs)
        # # print(model_input.shape)
        # model_input_tensor = torch.tensor(model_input).unsqueeze(0)
        tile_output = model(model_input_tensor).detach().numpy().squeeze()
        return tile_output
    run_name = os.path.basename(run_folder) # save this for plotting
    # !ls -l $run_folder/checkpoints
    resdepth_state_dict = torch.load(run_folder + "/checkpoints/Model_best.pth")
    print("Loaded best model from epoch", resdepth_state_dict['epoch'])
    model_args = {'n_input_channels': 3, 'start_kernel': 64, 'depth': 5, 'act_fn_encoder': 'relu', 'act_fn_decoder': 'relu', 'act_fn_bottleneck': 'relu', 'up_mode': 'transpose', 'do_BN': True, 'outer_skip': True, 'outer_skip_BN': False, 'bias_conv_layer': True}
    checkpoint_model = UNet(**model_args)
    checkpoint_model.load_state_dict(resdepth_state_dict["model_state_dict"])
    checkpoint_model.eval()
    print("loaded model from checkpoint")

    # TODO copied from previous experiment, curious
    dsm_mean_std = {'mean': None, 'std': 33.336839170631755}
    dsm_std = dsm_mean_std["std"]
    ortho_mean_std = {'mean': 261.1999816894531, 'std': 245.63670349121094}
    ortho_mean = ortho_mean_std["mean"]
    ortho_std = ortho_mean_std["std"]

    resdepth_model = ResDepthLM(model_args)
    trainer = pl.Trainer(check_val_every_n_epoch=5)
    trainer.fit(resdepth_model, dataloader, dataloader)# TODO change back to val))

    assert 0

    for b, batch in enumerate(dataloader):

        dsm_mean = batch["inputs"][:,0].mean()
        # tile_dsm_normalized = torchvision.transforms.Normalize(dsm_mean, dsm_std)(torch.tensor([tile_dsm.data])).numpy().squeeze()
        normalized_inputs = torch.tensor(batch["inputs"])
        normalized_inputs[:,0] =torchvision.transforms.Normalize(dsm_mean, dsm_std)(normalized_inputs[:,0])

        normalized_inputs[:,1:] = torchvision.transforms.Normalize(ortho_mean, ortho_std)(normalized_inputs[:,1:])

        output = call_model(checkpoint_model, normalized_inputs)
        print(output.shape)

        print("Plotting batch")

        # Plot example
        image = batch["inputs"]
        target = batch["target"]

        # print(image.shape)
        # print(target.shape)
        fig, ax = plt.subplots(BATCH_SIZE, 6, figsize=(15,15)) #TODO 2-3 inches high, 300dpi, padding bounds

        plt.suptitle(f"Batch of size {image.shape[0]}")
        print(batch["query"])
        assert 0

        for i in range(BATCH_SIZE):

            initial_dem = image[i][0].numpy().squeeze()
            ax[i][0].imshow(hillshade(initial_dem), cmap="gray", rasterized=True)
            im = ax[i][0].imshow(initial_dem, alpha=0.5, rasterized=True)
            ax[i][0].set_title("initial DEM")
            plt.colorbar(im, ax=ax[i][0], fraction=0.04)

            im = ax[i][1].imshow(image[i][1].numpy().squeeze(), cmap="gray", rasterized=True)
            ax[i][1].set_title("ortho left")
            plt.colorbar(im, ax=ax[i][1], fraction=0.04)

            im = ax[i][2].imshow(image[i][2].numpy().squeeze(), cmap="gray", rasterized=True)
            ax[i][2].set_title("ortho right")
            plt.colorbar(im, ax=ax[i][2], fraction=0.04)

            gt_dem = target[i].numpy().squeeze()
            ax[i][3].imshow(hillshade(gt_dem), cmap="gray", rasterized=True)
            im = ax[i][3].imshow(gt_dem, alpha=0.5, rasterized=True)
            ax[i][3].set_title("lidar DEM")
            plt.colorbar(im, ax=ax[i][3], fraction=0.04)

            output_dem = output[i]
            # Normalize the naive way
            output_dem = (output_dem * dsm_std) + dsm_mean.numpy()

            ax[i][4].imshow(hillshade(output_dem), cmap="gray", rasterized=True)
            im = ax[i][4].imshow(output_dem, alpha=0.5, rasterized=True)
            ax[i][4].set_title("refined DEM")
            plt.colorbar(im, ax=ax[i][4], fraction=0.04)

            im = ax[i][5].imshow(output_dem-initial_dem, cmap="RdBu", rasterized=True)
            ax[i][5].set_title("residual DEM refinement")
            plt.colorbar(im, ax=ax[i][5], fraction=0.04)




        plt.tight_layout()
        # plt.show()
        plt.savefig(f"rd_out_batch_{b}.png")


