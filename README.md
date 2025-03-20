# DeepDEM: Deep learning for stereo DEM refinement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14270049.svg)](https://doi.org/10.5281/zenodo.14270049)

This repository demonstrates the use of deep learning as an intelligent post-processing filter to refine digital surface models (DSMs) generated from stereo satellite imagery.

<p align="center">
<img src="docs/DeepDEM block diagram.png" width="800">
</p>

DeepDEM is a deep learning framework that uses a U-Net architecture with a ResNet encoder to perform residual refinement of an input DSM. The model accepts orthorectified stereo images and the corresponding initial DSM (along with other ancilliary inputs) and calculates residuals that minimize the L1 loss when comparing the initial DSM to ground truth DSM data. The model can also be trained on Digital Terrain Models (DTMs) to estimate corresponding DTM values beneath sparse vegetation.

## Overview of repository
This repository contains Jupyter notebooks and Python scripts needed to train and generate inferences from very-high-resolution (VHR) satellite imagery and initial DSM estimates using airborne lidar products. It is assumed that the user has access to stereo satellite imagery and is familar with stereo photogrammetry (we use the [Ames Stereo Pipeline (ASP) software package](https://stereopipeline.readthedocs.io/en/latest/introduction.html)) and lidar point cloud processing (we use [PDAL](https://pdal.io) to generate the inputs needed for DeepDEM.

## DeepDEM in action
<p align="center">
<img src="docs/fig1.png" width="800">
</p>

<p align="center">
<img src="docs/fig2.png" width="800">
</p>

These figures show examples of Maxar WorldView-1 satellite stereo images (far left), the intial stereo DSM prepared using ASP (second column top), and mask of valid pixels for terrain and conifer forests south of Mt. Baker in Washington state, USA. The model inferences (DSM and DTM shaded relief map) are shown in the third column. The last column shows the corresponding ground truth lidar DSM and DTM products used for validation, which were not seen by the model during training.

## Using this repository

### Installing dependencies
Once the repository is cloned, the `environment.yml` file can be used to create an conda environment with the required dependencies. We recommend using [`mamba`](https://github.com/conda-forge/miniforge) to speed up the install process. The environment is created and activated by the following commands from the root directory of the repository:

```
mamba env create -f environment.yml
...
mamba activate deep_dem
```

### Machine learning libraries
DeepDEM primarily uses [PyTorch-lightning](https://github.com/Lightning-AI/pytorch-lightning), [TorchGeo](https://github.com/microsoft/torchgeo), and [Segmentation Models for PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)

### Data prerequisites
It is expected that user has access to Level-1B panchromatic (single channel) stereo satellite imagery, which has been processed to produce a DEM and orthorectified images, along with a map of [triangulation error](https://stereopipeline.readthedocs.io/en/latest/correlation.html#triangulation-error). 
The `0_Download_LIDAR_data.ipynb` notebook can be used to download and process the [USGS 3DEP lidar data](https://www.usgs.gov/3d-elevation-program) to prepare the necessary DSM/DTM rasters for model training and validation. The ground truth for the results demonstrated here are obtained from the [3DEP LIDAR survey of Mt Baker](https://www.sciencebase.gov/catalog/item/58518b0ee4b0f99207c4f12c) acquired from 2015-08-26 to 2015-09-27.

### Notebooks
The notebooks are labeled in the sequence of execution for arbitrary inputs. The first cells contain file paths to the relevant datasets. 

#### 0 Data Preprocessing
Download 3DEP lidar (`0a_Download_LIDAR_data.ipynb`) and Harmonized Landsat-Sentinel (HLS) data (`0b_Download_HLS_data.ipynb`) for the area defined by the orthoimages and input stereo DEM.

The preprocessing (`0c_Data_Preprocessing.ipynb`) notebook reprojects all rasters to the same 3D CRS, prepares an NDVI raster from the HLS data, prepares a shared raster nodata mask and applies adaptive histogram normalization to the orthoimages. The rasters are now ready to be used by the model for training.

Global statistics for the full scene and a corresponding chip-size-dependent scaling factor are calculated in the `0d_Calculating_Scale_Factor.ipynb` notebook. These are used during model training to scale the input DSM values by the sample mean and a global scaling factor.  *Note: the output of this notebook should be used to manually override model defaults before training/inference (next section).*

#### 1 Model Training
The notebook `1a_TrainDeepDEM.ipynb` demonstrates training the DeepDEM model for the "standard" configuration with input channels for two stereo images, initial DSM, triangulation error map, NDVI map, and a nodata mask. 

#### 2 Model Inference
The notebook `2a_GenerateInferences.ipynb` demonstrates loading pre-trained model weights and running inferences. The inferences for individual tiles can be stitched together either using `rasterio` or the ASP `dem_mosaic` command-line utility, and code for both is provided. This notebook also demonstrates generating shaded relief maps using `gdaldem`.

### Scripts and modules
A more in-depth example of model training is provided under `scripts/1a_TrainDeepDEM.py`. This script shows how training parameters can be changed for various experiments, including model architecture, model inputs, and training hyperpameters. An example script to generate inferences is given in `scripts/1b_GenerateInferences.py`. This script requires the user to have a trained model, along with the appropriate inputs.

The code for the model dataloader is given in `scripts/dataset_modules.py`. This module defines `TorchGeo` derived classes that are used to define `Datasets` and `DataModules`. `Datasets` groups together raster files that comprise an area of study, making it easy to query spatial/temporal bounds, calculate raster statistics for each layer, and is a part of the internal plumbing to pass around data during training and inference. `DataModules` are a [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) idea implemented in `TorchGeo`, which encapsulates all of the methods needed to process data during model training - in this case setting up dataloaders and handing data movement between CPU/GPU.

`scripts/task_module.py` defines the `DeepDEMRegressionTask` class, which defines the DeepDEM model and all of the associated methods, such as `training_step`, `validation_step` and the primary method to pipe data through the model, `forward`. This class also defines the default data global scaling factors (`GSF_DICT`) which can be overridden by passing in a dictionary of values during model initialization.

### Support for other datasets
The full workflow can be used to train new models and generate inferences for arbitrary inputs, as long as all necessary input layers are available (orthorectified images, initial DSM, triangulation error). The user can use the numbered workflow of the Jupyter notebooks listed above and pipe the processed data through the model. If performing only inferences, the `1a_TrainDeepDEM.ipynb` notebook should be skipped, with only the path to weights from a trained model being provided in `1b_Generate_Inferences.ipynb`
=======
### Before training new models
It is recommended that users examine the _processed_ input rasters (e.g. plotting using matplotlib, loading in QGIS, calculate statistics) ahead of running the training script. This is to ensure that all of the datasets are correctly projected and aligned, and will serve as a sanity check before trying to train a model.

### Introducing new datasets
Code in this repository can be used to train new models, as well as generate inferences on datasets which contain the necessary input layers (orthorectified imagery, initial DSM estimate, triangulation errors). The user can use the numbered workflow of the Jupyter notebooks listed above and pipe the processed data through the model. If performing only inferences, the `1a_TrainDeepDEM.ipynb` notebook should be skipped, with only the path to weights from a trained model being provided in `1b_Generate_Inferences.ipynb`

### Trained models
DeepDEM model weights trained using the 2015-09-15 WorldView images and 3DEP LIDAR data for Mt. Baker, WA are available... 

Training metadata is stored along with the model weights in the checkpoint files within the `model_kwargs` dictionary. For example, the following code snippet shows the bands and path for training data for a specific model: 

```
>>> model = DeepDEMRegressionTask.load_from_checkpoint(path_to_checkpoint)
>>> print(f"Bands used for training: {model.model_kwargs['bands']} \Data path: {model.model_kwargs['datapath']}")

Bands used for training: ['asp_dsm', 'ortho_left', 'ortho_right', 'ndvi', 'nodata_mask', 'triangulation_error'] 
Data path: /mnt/working/karthikv/DeepDEM/data/mt_baker/WV01_20150911_1020010042D39D00_1020010043455300/processed_rasters
```

## Citation
See CITATION.cff

This work was supported by the NASA Decadal Survey Incubation program for the [Surface Topography and Vegetation (STV)](https://science.nasa.gov/earth-science/decadal-surveys/decadal-stv/) observable, grant #80NSSC22K1094 to the University of Washington.

## Additional notes
This work builds on the excellent [ResDepth](https://github.com/prs-eth/ResDepth/) work of [Stucker et al. (2022)](https://doi.org/10.1016/j.isprsjprs.2021.11.009). However, we extend the application to natural surfaces and vegetation, with airborne lidar data as a source of ground truth rather than CAD models. We demonstrate this approach to infer DTM output from satellite imagery and a DSM. We also explore the utility of additional input layers, self-consistent image and DSM co-registration, and improved network architectures. 
