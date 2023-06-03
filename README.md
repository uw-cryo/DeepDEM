# deep-elevation-refinement
ResDepth &amp; other deep learning approaches to improve DSMs/DTMs

# Note
This files in this repository will be restructured as the workflow for running experiments with different models has changed, so multiple layers of folders should be removed.

## Concepts to link to?
- ASP processing
- torchgeo
- PyTorch
- evaluation of outputs

# Instructions for training

## Installation
- `mamba env create --file environment.yml`
- Follow the ASP installation instructions to create a separate ASP installation (latest stable/nightly release; conda environment or download directly and add to path manually)

## Preparing the dataset
Follow the steps outlined in `dataset_processing/baker_csm/baker_csm_processing.ipynb` to start

TODO: Revised script version to work with new pairs 


## Training the network
Run train.py with modifications
- `python train.py`

Run with CLI (in progress):
- Check available parameter and model and dataset settings
- Define the configuration file as seen in `configs/example_config.yml`
- Run with commands like `python train_cli.py fit --data.train_directory="/path/to/training/data" --data.val_directory="/path/to/validation/data" --model.normalization="meanstd" ...`


## Inference and evaluation

Run inference from a checkpoint:
- `python inference.py tb_logs/resdepth_torchgeo/version_101/checkpoints/last.ckpt meanstd inference_v101_last_training_on_SCG.tif`
(This may change to model output postprocessing / statistics and figures script in combination with the LightningCLI `predict` subcommand)

Evaluation (besides inspection with QGIS or preferred raster workflow)
- Evaluation notebook that generates stats and figures coming soon


## Overview of key files
- `configs/` should store multiple experiment definitions that will be run & saved in logging folder
- `train_cli.py` and `train.py` are how to launch training runs
- `resdepth_lightning_module.py` wraps the model, optimizer, training and validation steps
- `tgdsm_lightning_data_module.py` wraps the training and validation DataLoaders
- `torchgeo_dataset.py` not required, but is base dataset implementation to feed stacks of aligned rasters to PyTorch
- `scg_stereo.sh` and stereo notebook- changing to generic stereo script or stick with notebook workflow?
- TODO Dataset inspection notebook revised `check_and_plot_tiles.ipynb` / `visualize_torchgeo_datasets.ipynb`
- environment.yml for future compatibility, keep versions as up-to-date as possible for dependencies like PyTorch, TorchGeo, Lightning, etc.

## How do I ...?
- Start a training run
- Add a new dataset
- Compile statistics about output
- Run trained model on some new input
- Change configuration of parameters/data/model/etc.
- Log metrics, or add my own?
- Add a new model, input, loss function, etc.
- Deal with temporal offset with my training data?
- ...

## Link to writeup and roadmap
- ... coming soon ...
