# This is the training script used to run all the experiments for the DeepDEM paper
# The experiments we run are: 
# 1. Effect of chip sizes on model training
# 2. Effect of excluding NDVI from model training
# 3. Using UNet vs UNet+ResNet architecture
# 4. Impact of using non-aligned ortho images (traditional processing)
# 5. Training on DTM vs DSM

from pathlib import Path
from torchgeo.datasets import stack_samples
import torch
from torch import nn 

from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as L
import kornia.augmentation as K

# local imports
from dataset_modules import CustomInputDataset, CustomDataModule
from task_module import DeepDEMRegressionTask

torch.set_float32_matmul_precision('medium')

BATCH_SIZE = 24
NUM_WORKERS = 16
CHANNEL_SWAP = True

transforms = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
)

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

# We calculate image parameters once before beginning training
datapath = '/mnt/working/karthikv/DeepDEM/data/09232024/WV01_20150911_1020010042D39D00_1020010043455300/processed_rasters'
tempdataset = CustomInputDataset(paths=datapath, bands=bands)
left_ortho_mean, left_ortho_std = tempdataset.compute_mean_std("ortho_left")
right_ortho_mean, right_ortho_std = tempdataset.compute_mean_std("ortho_right")

datamodule_params_template = {
    'dataset_class':CustomDataModule,
    'batch_size':BATCH_SIZE,
    'num_workers':NUM_WORKERS,
    'collate_fn':stack_samples,
    'cuda': torch.cuda.is_available(),
    'train_aug':transforms
}

model_kwargs_template = {
    'encoder_weights':'imagenet',
    'left_ortho_mean':left_ortho_mean,
    'left_ortho_std':left_ortho_std,
    'right_ortho_mean':right_ortho_mean,
    'right_ortho_std':right_ortho_std,
    'channel_swap':CHANNEL_SWAP,
    'do_BN':False,
    'bias_conv_layer':False,
    'lr':5e-4,
    'num_workers':NUM_WORKERS,
    'max_epochs':300,
    'lr_scheduler':False,
    'lr_scheduler_scale_factor':0.5,
    'lr_scheduler_patience':50,
    'early_stopping':True,
    'earlystopping_patience':60,
    'datapath':datapath
}

#############################
#############################
## Experiment 1
datapath = '/mnt/working/karthikv/DeepDEM/data/09232024/WV01_20150911_1020010042D39D00_1020010043455300/processed_rasters/'

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

MODEL_ENCODER = 'resnet18'
MODEL_TYPE = 'smp-unet'
CHIP_SIZES =  [128, 256, 512]
EXPERIMENT_NUMBER=1

for CHIP_SIZE in CHIP_SIZES:
    datamodule_params = datamodule_params_template.copy()
    datamodule_params['paths'] = datapath
    datamodule_params['chip_size'] = CHIP_SIZE
    datamodule_params['bands'] = bands
    datamodule = CustomDataModule(**datamodule_params)

    model_kwargs = model_kwargs_template.copy()
    model_kwargs['datapath'] = datapath
    model_kwargs['bands'] = bands
    model_kwargs['encoder'] = MODEL_ENCODER
    model_kwargs['model'] = MODEL_TYPE
    model_kwargs['chip_size'] = CHIP_SIZE
    task = DeepDEMRegressionTask(**model_kwargs)

    checkpoint_directory = Path(f'./checkpoints/experiment_group_{EXPERIMENT_NUMBER}')
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

    checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
    checkpoint_directory.mkdir(exist_ok=False)

    callbacks = [
                    LearningRateMonitor(logging_interval='step'), 
                    ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')
                ]
    if model_kwargs['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_loss", 
                                       min_delta=0.05, 
                                       patience=model_kwargs['earlystopping_patience'], 
                                       verbose=True, mode="min")) # type: ignore

    logger = TensorBoardLogger(save_dir="logs/", name=f"experiment_group_{EXPERIMENT_NUMBER}")

    trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                        default_root_dir=checkpoint_directory, 
                        max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                        log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                        callbacks=callbacks) # type: ignore

    trainer.fit(model=task, datamodule=datamodule)
    torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")

print(f"End of experiment {EXPERIMENT_NUMBER}")

#############################
#############################
## Experiment #2, removing NDVI

datapath = '/mnt/working/karthikv/DeepDEM/data/baker_csm_stack/processed_rasters/'

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    # "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

MODEL_ENCODER = 'resnet18'
MODEL_TYPE = 'smp-unet'
CHIP_SIZE = 256
EXPERIMENT_NUMBER=2

datamodule_params = datamodule_params_template.copy()
datamodule_params['paths'] = datapath
datamodule_params['chip_size'] = CHIP_SIZE
datamodule_params['bands'] = bands
datamodule = CustomDataModule(**datamodule_params)

model_kwargs = model_kwargs_template.copy()
model_kwargs['datapath'] = datapath
model_kwargs['bands'] = bands
model_kwargs['encoder'] = MODEL_ENCODER
model_kwargs['model'] = MODEL_TYPE
model_kwargs['chip_size'] = CHIP_SIZE
task = DeepDEMRegressionTask(**model_kwargs)

checkpoint_directory = Path(f'./checkpoints/experiment_group_{EXPERIMENT_NUMBER}')
checkpoint_directory.mkdir(exist_ok=True, parents=True)
model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
checkpoint_directory.mkdir(exist_ok=False)

callbacks = [
                LearningRateMonitor(logging_interval='step'), 
                ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')
            ]
if model_kwargs['early_stopping']:
    callbacks.append(EarlyStopping(monitor="val_loss", 
                                    min_delta=0.05, 
                                    patience=model_kwargs['earlystopping_patience'], 
                                    verbose=True, mode="min")) # type: ignore

logger = TensorBoardLogger(save_dir="logs/", name=f"experiment_group_{EXPERIMENT_NUMBER}")

trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                    default_root_dir=checkpoint_directory, 
                    max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                    log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                    callbacks=callbacks) # type: ignore

trainer.fit(model=task, datamodule=datamodule)
torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")

print(f"End of experiment {EXPERIMENT_NUMBER}")
# #############################
# #############################
# # Experiment #3, stock UNet in place of SMP-UNet
datapath = '/mnt/working/karthikv/DeepDEM/data/09232024/WV01_20150911_1020010042D39D00_1020010043455300/processed_rasters/'

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

MODEL_ENCODER = 'None'
MODEL_TYPE = 'unet'
CHIP_SIZE = 256
EXPERIMENT_NUMBER=3

datamodule_params = datamodule_params_template.copy()
datamodule_params['paths'] = datapath
datamodule_params['chip_size'] = CHIP_SIZE
datamodule_params['bands'] = bands
datamodule = CustomDataModule(**datamodule_params)

model_kwargs = model_kwargs_template.copy()
model_kwargs['datapath'] = datapath
model_kwargs['bands'] = bands
model_kwargs['encoder'] = MODEL_ENCODER
model_kwargs['model'] = MODEL_TYPE
model_kwargs['chip_size'] = CHIP_SIZE
task = DeepDEMRegressionTask(**model_kwargs)

checkpoint_directory = Path(f'./checkpoints/experiment_group_{EXPERIMENT_NUMBER}')
checkpoint_directory.mkdir(exist_ok=True, parents=True)
model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
checkpoint_directory.mkdir(exist_ok=False)

callbacks = [
                LearningRateMonitor(logging_interval='step'), 
                ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')
            ]
if model_kwargs['early_stopping']:
    callbacks.append(EarlyStopping(monitor="val_loss", 
                                    min_delta=0.05, 
                                    patience=model_kwargs['earlystopping_patience'], 
                                    verbose=True, mode="min")) # type: ignore

logger = TensorBoardLogger(save_dir="logs/", name=f"experiment_group_{EXPERIMENT_NUMBER}")

trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                    default_root_dir=checkpoint_directory, 
                    max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                    log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                    callbacks=callbacks) # type: ignore

trainer.fit(model=task, datamodule=datamodule)
torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")

print(f"End of experiment {EXPERIMENT_NUMBER}")

#############################
#############################
# Experiment #4 Traditional Stereo processed initial DSM
datapath = '/mnt/working/karthikv/DeepDEM/data/baker_traditional_processing/processed_rasters'

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_data",
]

MODEL_ENCODER = 'resnet18'
MODEL_TYPE = 'smp-unet'
CHIP_SIZE =  256
EXPERIMENT_NUMBER=4

datamodule_params = datamodule_params_template.copy()
datamodule_params['paths'] = datapath
datamodule_params['chip_size'] = CHIP_SIZE
datamodule_params['bands'] = bands
datamodule = CustomDataModule(**datamodule_params)

model_kwargs = model_kwargs_template.copy()
model_kwargs['datapath'] = datapath
model_kwargs['bands'] = bands
model_kwargs['encoder'] = MODEL_ENCODER
model_kwargs['model'] = MODEL_TYPE
model_kwargs['chip_size'] = CHIP_SIZE
task = DeepDEMRegressionTask(**model_kwargs)

checkpoint_directory = Path(f'./checkpoints/experiment_group_{EXPERIMENT_NUMBER}')
checkpoint_directory.mkdir(exist_ok=True, parents=True)
model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
checkpoint_directory.mkdir(exist_ok=False)

callbacks = [
                LearningRateMonitor(logging_interval='step'), 
                ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')
            ]
if model_kwargs['early_stopping']:
    callbacks.append(EarlyStopping(monitor="val_loss", 
                                    min_delta=0.05, 
                                    patience=model_kwargs['earlystopping_patience'], 
                                    verbose=True, mode="min")) # type: ignore

logger = TensorBoardLogger(save_dir="logs/", name=f"experiment_group_{EXPERIMENT_NUMBER}")

trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                    default_root_dir=checkpoint_directory, 
                    max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                    log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                    callbacks=callbacks) # type: ignore

trainer.fit(model=task, datamodule=datamodule)
torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")

print(f"End of experiment {EXPERIMENT_NUMBER}")

#############################
#############################
## Experiment #5 Using DTM inplace of DSM
datapath = '/mnt/working/karthikv/DeepDEM/data/09232024/WV01_20150911_1020010042D39D00_1020010043455300/processed_rasters_dtm'

bands = [
    "asp_dsm",
    "ortho_left",
    "ortho_right",
    "ndvi",
    "nodata_mask",
    "triangulation_error",
    "lidar_dtm_data",
]

MODEL_ENCODER = 'resnet18'
MODEL_TYPE = 'smp-unet'
CHIP_SIZE =  256
EXPERIMENT_NUMBER=5

datamodule_params = datamodule_params_template.copy()
datamodule_params['paths'] = datapath
datamodule_params['chip_size'] = CHIP_SIZE
datamodule_params['bands'] = bands
datamodule = CustomDataModule(**datamodule_params)

model_kwargs = model_kwargs_template.copy()
model_kwargs['datapath'] = datapath
model_kwargs['bands'] = bands
model_kwargs['encoder'] = MODEL_ENCODER
model_kwargs['model'] = MODEL_TYPE
model_kwargs['chip_size'] = CHIP_SIZE
task = DeepDEMRegressionTask(**model_kwargs)

checkpoint_directory = Path(f'./checkpoints/experiment_group_{EXPERIMENT_NUMBER}')
checkpoint_directory.mkdir(exist_ok=True, parents=True)
model_count = len([x for x in list(checkpoint_directory.glob('*')) if x.is_dir()]) + 1

checkpoint_directory = checkpoint_directory / f"version_{str(model_count).zfill(3)}"
checkpoint_directory.mkdir(exist_ok=False)

callbacks = [
                LearningRateMonitor(logging_interval='step'), 
                ModelCheckpoint(dirpath=checkpoint_directory, monitor='val_loss', mode='min')
            ]
if model_kwargs['early_stopping']:
    callbacks.append(EarlyStopping(monitor="val_loss", 
                                    min_delta=0.05, 
                                    patience=model_kwargs['earlystopping_patience'], 
                                    verbose=True, mode="min")) # type: ignore

logger = TensorBoardLogger(save_dir="logs/", name=f"experiment_group_{EXPERIMENT_NUMBER}")

trainer = L.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu", 
                    default_root_dir=checkpoint_directory, 
                    max_epochs=model_kwargs['max_epochs'], logger=logger, check_val_every_n_epoch=1, # type: ignore
                    log_every_n_steps=1, fast_dev_run=False, # set fast_dev_run to True for sanity check (dummy run) before training 
                    callbacks=callbacks) # type: ignore

trainer.fit(model=task, datamodule=datamodule)
torch.save(task.model.state_dict(), checkpoint_directory/f"model_weights_version{model_count}.pth")

print(f"End of experiment {EXPERIMENT_NUMBER}")