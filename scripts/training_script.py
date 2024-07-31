
# %%
import sys
from pathlib import Path
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datasets import stack_samples
from torchgeo.samplers import RandomBatchGeoSampler
import torch
from torch import nn 

from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as L
import kornia.augmentation as K

# %%
# local imports
# sys.path.insert(0, str(Path('.').absolute().parent/'scripts'))
from dataset_modules import CustomInputDataset, CustomDataModule
from task_module import DeepDEMRegressionTask

# %%

datapath = '/mnt/1.0_TB_VOLUME/karthikv/DeepDEM/data/baker_csm_stack/processed_rasters/'

CHIP_SIZES =  [128, 256, 512] #  128, 
BATCH_SIZE = 8
NUM_WORKERS = 8


bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

transforms = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
)

for CHIP_SIZE in CHIP_SIZES:
    datamodule_params = {
        'paths': datapath,
        'dataset_class':CustomDataModule,
        'chip_size':CHIP_SIZE,
        'batch_size':BATCH_SIZE,
        'num_workers':NUM_WORKERS,
        'collate_fn':stack_samples,
        'cuda': torch.cuda.is_available(),
        'bands':bands,
        'train_aug':transforms
    }
    datamodule = CustomDataModule(**datamodule_params)

    # %%

    tempdataset = CustomInputDataset(paths=datapath, bands=bands)
    left_ortho_mean, left_ortho_std = tempdataset.compute_mean_std("ortho_left")
    right_ortho_mean, right_ortho_std = tempdataset.compute_mean_std("ortho_right")

    model_kwargs = {
        'model':'smp-unet',
        'encoder':'resnet18',
        'encoder_weights':'imagenet',
        'bands':bands,
        'left_ortho_mean':left_ortho_mean,
        'left_ortho_std':left_ortho_std,
        'right_ortho_mean':right_ortho_mean,
        'right_ortho_std':right_ortho_std,
        'chip_size':CHIP_SIZE,
        'do_BN':False,
        'bias_conv_layer':False,
        'lr':5e-4,
        'num_workers':NUM_WORKERS,
        'max_epochs':100,
        'lr_scheduler':True,
        'lr_scheduler_scale_factor':0.5,
        'lr_scheduler_patience':10
    }
    task = DeepDEMRegressionTask(**model_kwargs)

    # %%
    checkpoint_directory = Path(f'./checkpoints/checkpoint_directory_{datetime.now().strftime("%Y%m%d")}')
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

    checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
    checkpoint_directory.mkdir(exist_ok=False)

    callbacks =[LearningRateMonitor(logging_interval='step'), ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')]
    logger = TensorBoardLogger(save_dir="logs/", name=f"my_experiment_{datetime.now().strftime("%Y%m%d")}")


    # %%
    trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                        default_root_dir=checkpoint_directory, 
                        max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                        log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                        callbacks=callbacks) # type: ignore

    # %%
    trainer.fit(model=task, datamodule=datamodule)

    torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")