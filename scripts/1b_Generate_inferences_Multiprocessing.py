# %% [markdown]
# This notebook demonstrates how to generate inference for a scene using a previously trained DeepDEM model

# %%
# rasterio imports
import rasterio
from rasterio.merge import merge

from rasterio import transform

# torchgeo imports
from torchgeo.samplers import GridGeoSampler
# from torchgeo.datasets import RasterDataset

# pytorch imports 
import torch

# misc imports
from functools import reduce, partial
# import operator
from pathlib import Path, PurePath
import numpy as np

import yaml
from pathlib import Path
import subprocess

from multiprocessing import Pool

# local imports
import sys
sys.path.insert(0, str(Path('.').absolute().parent/'scripts'))

from task_module import DeepDEMRegressionTask
from dataset_modules import CustomInputDataset, CustomDataModule
from torchgeo.datasets import stack_samples
from torchgeo.samplers import BatchGeoSampler
from torch import nn 
from functools import partial


# %%
model_path= Path('/mnt/working/karthikv/DeepDEM/scripts/checkpoints/experiment_group_1/version_001/')
model_checkpoint = list(model_path.glob('*.ckpt'))[0]
model = DeepDEMRegressionTask.load_from_checkpoint(model_checkpoint).cuda().eval();
# model.model_kwargs['channel_swap'] = False

bands, datapath, chip_size = model.model_kwargs['bands'], model.model_kwargs['datapath'], model.model_kwargs['chip_size']
bands.remove('lidar_data')
inference_dataset = CustomInputDataset(datapath, bands=bands)
data_sampler = GridGeoSampler(inference_dataset, size=chip_size, stride=chip_size//2)

# %%
with rasterio.open(inference_dataset.files[0]) as ds:
    template_profile = ds.profile

# %%
def generate_write_inference(index, samples, chip_size, dataset, model, output_path, template_profile):
    
    if not isinstance(output_path, PurePath):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    img = []
    bounds = []    
    template_profile.update({
        'width':chip_size,
        'height':chip_size
        })

    # manually create a batch of images
    for sample in samples:
        item = dataset.__getitem__(sample)
        img.append(item['image'])
        bounds.append((sample.minx, sample.miny, sample.maxx, sample.maxy))
    
    # perform inference on a batch of images
    inference = model.forward(torch.stack(img, dim=0).cuda(), stage='inference').cpu().detach().numpy()
    
    # calculate transform for each tile and write it out
    for i in range(inference.shape[0]):
        template_profile.update({
            'transform':transform.from_bounds(*bounds[i], chip_size, chip_size)
        })

        with rasterio.open(output_path / f"inference_{str(index+i).zfill(7)}.tif", 'w', **template_profile) as ds:
            ds.write(inference[i].reshape(1, *inference[i].shape))


model_name = '_'.join(str(model_path).split('/')[-2:])
output_path = Path(f'../outputs/{model_name}')

generate_write_inference = partial(generate_write_inference, 
                                   chip_size=chip_size,
                                   model=model,
                                   dataset=inference_dataset,
                                   output_path=output_path,
                                   template_profile=template_profile)

# %%
img = []
BATCH_SIZE = 32
data_sampler = list(data_sampler)
for i in range(len(data_sampler)):
    generate_write_inference(index=i, samples=data_sampler[i:i+BATCH_SIZE])
    break

# %%
inference = model.forward(torch.stack(img, dim=0).cuda(), stage='inference').cpu().detach().numpy()

# %%
data_sampler = list(data_sampler)

# %%
# Chunk samples into batches for parallel processing
BATCH_SIZE = 32

for i in range(0, len(data_sampler), BATCH_SIZE):
    print(data_sampler[i:i+BATCH_SIZE])
    

# %%
len(data_sampler)

# %%
# Generate inferences
for i, sample in enumerate(data_sampler):
    generate_write_inference(i, sample)

# %%
inferences = sorted(list(output_path.glob('inference_*.tif')))[::-1]
merged_inference, merge_transform = merge(inferences)

# %%
inference_profile = template_profile.copy()
inference_profile['height'] = merged_inference.shape[-2]
inference_profile['width'] = merged_inference.shape[-1]
inference_profile['transform'] = merge_transform

with rasterio.open(output_path.parent / f'{output_path.name}.tif', 'w', **inference_profile) as ds:
    ds.write(merged_inference)

# %%
subprocess.run(["gdaldem", "hillshade", "-compute_edges", output_path.parent / f'{output_path.name}.tif', output_path.parent / f'{output_path.name}_hs.tif'])


