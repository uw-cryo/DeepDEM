# Code defining the model for DeepDEM

# The DeepDEM framework uses a UNet for DEM refinement
from UNet import UNet

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
from kornia.enhance import normalize


class DeepDEMRegressionTask(BaseTask):
    """
    Model class to refine DSM generated from stereo images
    """

    # global scaling factor for DSM
    GSF_DICT = {
        64: tensor(9.88),
        128: tensor(18.71),
        256: tensor(35.16),
        512: tensor(64.32),
        1024: tensor(111.88),
    }

    def __init__(
        self,
        **model_kwargs: dict,
    ):
        self.model_kwargs = model_kwargs
        assert set(["model", "bands"]).issubset(
            set(self.model_kwargs.keys())
        ), "keyword arguments must include 'model' and 'bands'"
        self.n_channels = (
            len(self.model_kwargs["bands"]) - 1
        )  # the lidar data is not part of the inputs

        # this will remain constant in an experiment, for a given chip size
        self.gsf = self.GSF_DICT[self.model_kwargs["chip_size"]]

        super().__init__()
        self.lr_scheduler = self.model_kwargs['lr_scheduler']
        self.lr_scheduler_scale_factor = self.model_kwargs['lr_scheduler_scale_factor']
        self.lr_scheduler_patience = self.model_kwargs['lr_scheduler_patience']

        self.left_ortho_mean = tensor(self.model_kwargs["left_ortho_mean"])
        self.left_ortho_std = tensor(self.model_kwargs["left_ortho_std"])
        self.right_ortho_mean = tensor(self.model_kwargs["right_ortho_mean"])
        self.right_ortho_std = tensor(self.model_kwargs["right_ortho_std"])

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
        # print("Types: ", type(y), type(y_hat))
        loss = nn.functional.l1_loss(y, y_hat, reduction="none")

        # print("calculated loss: ", (loss * mask).sum() / mask.sum())

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
            for i in range(img.shape[0]):  # iterate over all batch samples
                sample_mean = img[i, index, ...].mean()
                img[i, index, ...] = normalize(
                    img[i, index, ...], mean=sample_mean, std=self.gsf
                )
        except ValueError:
            # if an initial dsm is not provided, we assume zero values
            initial_dsm = torch.zeros_like(img[:, 0, ...].squeeze())

        # scale ortho_left image
        try:
            index = self.model_kwargs["bands"].index("ortho_left")
            img[:, index, ...] = normalize(
                img[:, index, ...], self.left_ortho_mean, self.left_ortho_std
            )
        except ValueError:
            pass

        # scale ortho_right image
        try:
            index = self.model_kwargs["bands"].index("ortho_right")
            img[:, index, ...] = normalize(
                img[:, index, ...], self.right_ortho_mean, self.right_ortho_std
            )
        except ValueError:
            pass

        # print("channel wise min max: ")
        # for i in range(img.shape[1]):
        #     print(img[:, i, ...].min(), img[:, i, ...].max())

        output = self.model.forward(img).squeeze()

        # print("Output shape in forward: ", output.shape)
        for i in range(output.shape[0]):  # type:ignore
            # print(output[i, ...].min(), output[i, ...].max(), initial_dsm[i, ...].min(), initial_dsm[i, ...].max(), self.gsf)
            output[i, ...] = output[i, ...] * self.gsf + initial_dsm[i, ...]  # type: ignore

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
        batch = args[0]
        x = batch["image"]
        x, y = (
            x[:, :-1, ...],
            x[:, -1, ...].squeeze(),
        )  # the lidar data is the last channel
        # print("Type in training step: ", type(x), type(y))
        batch_size = x.shape[0]
        y_hat = self.forward(x).squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, *args, **kwargs):
        """Validation step"""
        batch = args[0]
        x = batch["image"]
        x, y = x[:, :-1, ...], x[:, -1, ...].squeeze()
        batch_size = x.shape[0]
        y_hat = self.forward(x).squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.log("val_loss", loss, batch_size=batch_size)

    def test_step(self, *args, **kwargs):
        """Test step"""
        batch = args[0]
        x = batch["image"]
        x, y = x[:, :-1, ...], x[:, -1, ...].squeeze()
        batch_size = x.shape[0]
        y_hat = self.forward(x).squeeze()
        mask = self.return_batch_mask(x)
        loss: Tensor = self.model_loss(y_hat, y, mask)
        self.log("test_loss", loss, batch_size=batch_size)

    # def predict(self, *args, **kwargs):
    #     '''Return model inferences'''
    #     batch = args[0]  # noqa: F841
    #     return

    #     retu
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configuring optimizers"""
        optimizer = optim.Adam(self.parameters(), lr=self.model_kwargs['lr'])

        # Define scheduler
        scheduler = None
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.lr_scheduler_scale_factor,
                patience=self.lr_scheduler_patience, verbose=True, threshold=0.1
            )

        self.scheduler = scheduler
        self.optimizer = optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
