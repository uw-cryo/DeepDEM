# DeepDEM
Satellite stereo DEM refinement using deep learning

# Note
This files in this repository will be restructured as the workflow for running experiments with different models has changed, so multiple layers of folders should be removed.

## Useful concepts to familiarize oneself with first
- [Demo notebook](https://github.com/uw-cryo/DeepDEM/blob/main/notebooks/deepDEM_demo.ipynb)
- Ames Stereo Pipeline (ASP) and very-high-resolution (VHR) satellite stereo processing
- [torchgeo](https://torchgeo.readthedocs.io/en/stable/)
- [ResDepth](https://github.com/prs-eth/ResDepth/)
- PyTorch / [PyTorch Lightning](https://lightning.ai/)

# Instructions for training

## Installation
- Run `mamba env create --file environment.yml`
- Follow the ASP installation instructions to create a separate ASP installation (latest stable/nightly release; conda environment or download directly and add to path manually) as conda environment is not compatible

## Preparing the input datasets
Follow the steps outlined in `dataset_processing/baker_csm/baker_csm_processing.ipynb` to start

TODO: Integrate revised scripts to work with new pairs 

## Training the network
Run `train.py` with modifications:
- `python train.py`
Run with LightningCLI (in progress, encouraged):
- Check available parameter and model and dataset settings
- Run with commands like `python train_cli.py fit --data.train_directory="/path/to/training/data" --data.val_directory="/path/to/validation/data" --model.normalization="meanstd" ...`

## Inference and evaluation

Run inference from a checkpoint:
- `python inference.py tb_logs/resdepth_torchgeo/version_101/checkpoints/last.ckpt meanstd inference_v101_last_training_on_SCG.tif`
(This may change to model output postprocessing / statistics and figures script in combination with the LightningCLI `predict` subcommand)

Evaluation (besides inspection with QGIS or preferred raster workflow)
- Evaluation notebook that generates stats and figures coming soon

## Overview of key files
- Data directory (outside the repository) contains
    - **IMPORTANT NOTE**: Used Symlinks to mosaic, usgs_all616_laz_filtered_dem_mask_nlcd_rock_exclude_glaciers, shashank_data, other data files that are too large to live in the repository!
    - Experiment logs & model checkpoints (tb_logs/)
- `.gitignore` excludes most types geospatial files from showing up in git but tracking some of these is important, for example reference DEM rasters used in stereo processing are included.
- `configs/` should store multiple experiment definitions that will be run & saved in logging folder
- `train_cli.py` and `train.py` are how to launch training runs
    - train_cli.py with default arguments `python train_cli.py fit --print_config` is not as helpful as looking at the example YAML file for Mt Baker training/validation split in `configs/`
- `resdepth_lightning_module.py` wraps the model, optimizer, training and validation steps
- `tgdsm_lightning_data_module.py` wraps the training and validation DataLoaders
- `torchgeo_dataset.py` is not required moving forward, but is base dataset implementation to feed stacks of aligned rasters to PyTorch
    - This implementation could be modified to remove hardcoded filenames for the different layers
- `scg_stereo.sh` and stereo notebook- changing to generic stereo script or stick with notebook workflow?
- TODO Dataset inspection notebook revised `check_and_plot_tiles.ipynb` / `visualize_torchgeo_datasets.ipynb`
- environment.yml for future compatibility, keep versions as up-to-date as possible for dependencies like PyTorch, TorchGeo, Lightning, etc.

## Suggestions for a fresh start
- With all the updates to torchgeo and pytorch-lightning that appeared since the beginning of the project it is now easier to deal with handling stacks of multiple datasets and also dealing with many possible configuration parameters, etc.
- For multiple datasets and combinations of layers it will be important to pull that sort of configuration out of code files and into the YAML configs. Modules will have to be rewritten to normalize the correct layer given its index in the tensor and desired method/parameters, etc.

## How do I ...?
### Start a training run

### Add a new dataset
- Find your lidar or other desired training target (assuming point cloud format, but workflow should be compatible with rasters). For Mount Baker 2015 we downloaded, filtered, and merged the USGS 1km x 1km point clouds (USGS Entwine is likely a simpler source), and with South Cascade 2019 the starting point was a single point cloud.
- Set up stereo directory with L1B images and camera model files (e.g. DigitalGlobe `.r100.xml`) files.

### Train a model
* Current way:
* Edit the code and set up the desired dataset, parameters to use, model, and so on, then:
* `python train.py path/to/previous/checkpoint.ckpt`
* New and better way in progress to pull configuration choices out of the code:
* `python train_cli.py fit -c path/to/config.yaml`
* Define the configuration file as seen in `configs/example_config.yml` and/or a combination

### View logs in tensorboard
For experiments <= 145 generated with `train.py`: use `tensorboard serve --logdir path/to/data/torchgeo_experiments/tb_logs`
Experiments with > 145 will likely be saved in `lightning_logs`

### Run trained model on some new input
- Find checkpoints here on Google Drive TODO paste new link
- Note that these were trained for a specific set of inputs (in order) and so serve as a starting point. It might be possible to copy weights & add channels

### Evaluate stereo processing and refinement products in QGIS?
- Download the desired layers and set up appearance under Symbology
- A helpful way to determine which areas were used for training and which were validation/test patches: use a shapefile geojson (similar to those provided by USGS for their lidar tiles), or create a binary mask from a mosaiced validation dataset raster (using gdal_merge.py or similar approach) to overlay with different colors.
- For input/output DEMs and DEM differences, set resampling to Cubic for viewing purposes
- Hillshade - choose between GDAL (`gdaldem hillshade`) and QGIS choice
- For difference maps: in Symbology set Min and Max e.g. -1 m and +1 m, Color ramp red to blue
- Orthoimages may need to be stretched / range limited depending on the goal.
- Triangulation error with a small maximum value is helpful to indicate hotspots of larger intersection errors
- Copy and Paste Styles as desired to get consistent appearance & scales across layers
- It is useful to include other layers for context like the rasterized lidar timestamps and land cover classes

More:
- Compile statistics about output
- Change configuration of parameters/data/model/etc.
- Log metrics, or add my own?
- Add a new model, input, loss function, etc.
- Deal with temporal offset with my training data?
- ...

## Link to writeup and roadmap
- ... coming soon ...
