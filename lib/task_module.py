# Code defining the model for DeepDEM

# The DeepDEM framework uses a UNet for DEM refinement
from UNet import UNet
# from UNet import UNet

# We can also use UNets with varying encoders
from segmentation_models_pytorch import Unet as smp_unet

# pytorch imports
from torch import optim, nn, Tensor, tensor
import torch

# pytorch-lightning imports
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import lightning as L

# torchgeo imports
from torchgeo.trainers import BaseTask

# misc imports
import numpy as np
import random
from kornia.enhance import normalize


class DeepDEMRegressionTask(BaseTask):
    """
    Model class to refine DSM generated from stereo images
    """

    # global scaling factor for DSM
    # These are calculated from the 2015 Mt Baker WV1 images (0b_Calculating_Scale_Factor.ipynb)
    # These values can be overridden by providing a dictionary containing
    # chip size:scaling factor key value pairs when initializing the model
    GSF_DICT = {
        64: tensor(9.86),
        128: tensor(18.59),
        256: tensor(34.90),
        512: tensor(63.51),
        1024: tensor(112.11),
    }

    def __init__(
        self,
        **model_kwargs: dict,
    ):
                
        if 'GSF_DICT' not in model_kwargs:
            print("DeepDEMRegressionTask: *** Loading global scaling factors for WV1 Mt Baker data ***")
            print("DeepDEMRegressionTask: *** This can be overridden when initializing model (DeepDEMRegressionTask) ***")
            # this will remain constant in an experiment, for a given chip size
        else:
            self.GSF_DICT = model_kwargs['GSF_DICT']
        
        self.gsf = self.GSF_DICT[model_kwargs["chip_size"]]

        self.save_hyperparameters(ignore=['transforms'])

        self.model_kwargs = model_kwargs
        assert set(["model", "bands"]).issubset(
            set(self.model_kwargs.keys())
        ), "keyword arguments must include 'model' and 'bands'"
        self.n_channels = (
            len(self.model_kwargs["bands"]) - 1
        )  # the lidar data is not part of the inputs

        self.lr_scheduler = self.model_kwargs['lr_scheduler']
        self.lr_scheduler_scale_factor = self.model_kwargs['lr_scheduler_scale_factor']
        self.lr_scheduler_patience = self.model_kwargs['lr_scheduler_patience']

        self.train_losses = []
        self.val_losses = []
        super().__init__()

    def configure_models(self):
        """Initialize and configure model"""

        assert self.model_kwargs["model"] in [
            "unet",
            "smp-unet",
        ], "model must be either 'unet' or 'smp-unet'"
        if self.model_kwargs["model"] == "unet":
            self.model = UNet(
                n_input_channels=self.n_channels,
                depth=int(np.log2(self.model_kwargs["chip_size"])),
                do_BN=self.model_kwargs["do_BN"],
                bias_conv_layer=self.model_kwargs["bias_conv_layer"],
            )
        else:
            self.model = smp_unet(
                self.model_kwargs["encoder"],
                encoder_weights=self.model_kwargs["encoder_weights"],
                activation=None,
                in_channels=self.n_channels,
            )

    def model_loss(self, y_hat, y, mask):
        """
        The model uses L1 loss
        We exclude no data regions from the loss calculation
        """
        loss = nn.functional.l1_loss(y, y_hat, reduction="none")

        return (loss * mask).sum() / mask.sum()

    def forward(self, img, **kwargs):
        """
        Forward pass of data through model
        This requires the ortho images and the initial DSM to be scaled, if present
        The output of the model also needs to be scaled before returning to compute losses
        """
        # scale the asp_dem
        try:
            index = self.model_kwargs["bands"].index("asp_dsm")
            initial_dsm = img[:, index, ...].detach().clone()
            sample_means = img[:, index, ...].mean(dim=(1, 2), keepdim=True)
            img[:, index, ...] = (img[:, index, ...] - sample_means)/self.gsf
        except ValueError:
            # if an initial dsm is not provided, we assume zero values
            initial_dsm = torch.zeros_like(img[:, 0, ...].squeeze())
        
        # left/right channel swap
        if (self.model_kwargs['channel_swap']) and (kwargs['stage']=='train'):
            try:
                index1, index2 = self.model_kwargs["bands"].index("ortho_channel1"), self.model_kwargs["bands"].index("ortho_channel2")
                _left, _right = img[:, index1, ...], img[:, index2, ...]
                _left_new, _right_new = _left.detach().clone(), _right.detach().clone()
                for i in range(img.shape[0]):
                    if random.random() > 0.5:
                        _left_new[i] = _right[i]
                        _right_new[i] = _left[i]


                img[:, index1, ...] = _left_new
                img[:, index2, ...] = _right_new
            except ValueError:
                pass

        output = self.model.forward(img).squeeze()
        output = output*self.gsf + initial_dsm

        return output

    def return_batch_mask(self, img):
        """Return nodata mask channel if present in input data"""
        mask = torch.ones_like(img[:, 0, ...])

        try:
            index = self.model_kwargs["bands"].index("nodata_mask")
            mask = img[:, index, ...]
        except ValueError:
            pass

        return mask

    def training_step(self, *args, **kwargs):
        """Training step"""
        x = args[0]
        x, y = (
            x[:, :-1, ...],
            x[:, -1, ...].squeeze(),
        )  # the lidar data is the last channel
        batch_size = x.shape[0]
        y_hat = self.forward(x, stage='train').squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.log("train_loss", loss)
        self.train_losses.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_epoch_loss", torch.tensor(self.train_losses).sum()/len(self.train_losses))
        self.train_losses = []

    def validation_step(self, *args, **kwargs):
        """Validation step"""
        x = args[0]
        x, y = x[:, :-1, ...], x[:, -1, ...].squeeze()
        batch_size = x.shape[0]
        y_hat = self.forward(x, stage='validation').squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.val_losses.append(loss)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_epoch_loss", torch.tensor(self.val_losses).sum()/len(self.val_losses))
        self.val_losses = []

    def test_step(self, *args, **kwargs):
        """Test step"""
        x = args[0]
        x, y = x[:, :-1, ...], x[:, -1, ...].squeeze()
        batch_size = x.shape[0]
        y_hat = self.forward(x, stage='test').squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.log("test_loss", loss, batch_size=batch_size)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configuring optimizers"""
        
        optimizer_scheduler_dict = {}

        self.optimizer = optim.Adam(self.parameters(), lr=self.model_kwargs['lr'])
        optimizer_scheduler_dict.update({
            'optimizer':self.optimizer
        })

        # Define scheduler
        self.scheduler = None
        if self.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.lr_scheduler_scale_factor,
                patience=self.lr_scheduler_patience, verbose=True, threshold=0.1,
                min_lr=1e-6
            )

            optimizer_scheduler_dict.update({
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val_epoch_loss",
                    "frequency": 1,
                },                
            })

        
        return optimizer_scheduler_dict