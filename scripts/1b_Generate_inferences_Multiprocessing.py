# This notebook demonstrates how to generate inference for a scene using a previously trained DeepDEM model

# rasterio imports
import rasterio
from rasterio.merge import merge
from rasterio import transform

# torchgeo imports
from torchgeo.samplers import GridGeoSampler

# pytorch imports 
import torch
import torch.multiprocessing as mp

# misc imports
# from functools import partial
from pathlib import Path # , PurePath
import subprocess
from multiprocessing import Pool
import time

# local imports
from task_module import DeepDEMRegressionTask
from dataset_modules import CustomInputDataset

def generate_write_inference(index, samples, chip_size, model, dataset, output_path, template_profile):
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

def main():
    model_path= Path('/mnt/working/karthikv/DeepDEM/scripts/checkpoints/experiment_group_1/version_001/')
    model_checkpoint = list(model_path.glob('*.ckpt'))[0]
    model = DeepDEMRegressionTask.load_from_checkpoint(model_checkpoint).cuda().eval();

    bands, datapath, chip_size = model.model_kwargs['bands'], model.model_kwargs['datapath'], model.model_kwargs['chip_size']
    bands.remove('lidar_data')
    inference_dataset = CustomInputDataset(datapath, bands=bands)
    data_sampler = GridGeoSampler(inference_dataset, size=chip_size, stride=chip_size//2)

    with rasterio.open(inference_dataset.files[0]) as ds:
        template_profile = ds.profile

    model_name = '_'.join(str(model_path).split('/')[-2:])
    output_path = Path(f'../outputs/{model_name}')
    if not output_path.exists():
        output_path.mkdir()

    BATCH_SIZE = 32
    data_sampler = list(data_sampler)
    multiprocessing_args = [(
        i, 
        data_sampler[i:i+BATCH_SIZE],
        chip_size,
        model,
        inference_dataset,
        output_path,
        template_profile
        ) 
        for i in range(0, len(data_sampler), BATCH_SIZE)]
    start_time = time.time()
    mp.set_start_method('spawn')
    with mp.Pool(processes=4) as pool: # mp.cpu_count()
        pool.starmap(generate_write_inference, multiprocessing_args)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Inference run time: {run_time:.4f}")
    
    
    inferences = sorted(list(output_path.glob('inference_*.tif')))[::-1]
    start_time = time.time
    merged_inference, merge_transform = merge(inferences)
    end_time = time.time
    run_time = end_time - start_time
    print(f"Merge run time: {run_time:.4f}")

    inference_profile = template_profile.copy()
    inference_profile['height'] = merged_inference.shape[-2]
    inference_profile['width'] = merged_inference.shape[-1]
    inference_profile['transform'] = merge_transform

    # Write out merged inference
    with rasterio.open(output_path.parent / f'{output_path.name}.tif', 'w', **inference_profile) as ds:
        ds.write(merged_inference)

    # Create hillshaded raster using GDAL
    subprocess.run(["gdaldem", "hillshade", "-compute_edges", output_path.parent / f'{output_path.name}.tif', output_path.parent / f'{output_path.name}_hs.tif'])

if __name__ == '__main__':
    main()
