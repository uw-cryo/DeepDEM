# DeepDEM
This repository demonstrates the use of deep learning as an intelligent post-processing filter in refining digital surface models generated from satellite imagery.

<p align="center">
<img src="docs/DeepDEM block diagram.png" width="800">
</p>

### Overview of repository
This repository contains code in the form of Jupyter notebooks and Python scripts needed to train and generate inferences from satellite imagery and initial DSM estimates. It is assumed that the user has access to stereo satellite imagery and is familar with stereogrammetry tools (we use the [Ames Stereo Pipeline software package](https://stereopipeline.readthedocs.io/en/latest/introduction.html)) to generate the inputs needed for DeepDEM.

#### Notebooks

#### Scripts and modules

### Installing dependencies
The included environment file can be used to create an conda environment containing dependencies needed to run the codes in this repository. We recommend using [`mamba`](https://github.com/conda-forge/miniforge) to speed up the install process. The environment is created and activated by the following commands from the root directory of the repository:

```
mamba env create -f environment.yml
...
mamba activate deep_dem
```

### Training models

### Generating inferences

### Introducing new datasets
Code in this repository can be used to train new models, as well as genenrate inferences on datasets which contain the necessary input layers (orthorectified imagery, initial DSM estimate, triangulation errors)

### Trained models

### Sample outputs

# Paper


### Additional notes
This work is based on the [ResDepth](https://github.com/prs-eth/ResDepth/) work of Stucker et al. (2021), but extends the application to vegetation and natural surfaces and uses LIDAR data as a source of ground truth in place of well defined CAD models. The work here also demonstrates using this approach to generate Digital Terrain Models (DTM) from satellite imagery and and an initially derived DSM. We also explore the utility of additional input layers, self consistent orthorectification, and improved network architectures towards refining initial estimates of DSMs generated using photogrammetry.