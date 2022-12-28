import sys

import torch
import torchvision
from pytorch_lightning import LightningModule

# TODO figure out how to change to relative import of ResDepth UNet
sys.path.insert(0, "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/ResDepth/")
from lib.UNet import UNet

from train_utils import dsm_std, ortho_mean, ortho_std, trierror_mean, trierror_std


class TGDSMLightningModule(LightningModule):
    """Module to train ResDepth with PyTorch Lightning"""

    def __init__(
        self,
        n_input_channels,
        lr=0.00002,
        weight_decay=1e-5,
        checkpoint=None,
        loss_fn=torch.nn.L1Loss,
        model=None,
        model_args=None,
        normalization="minmax",
    ):
        """
        n_input_channels 3 or 4 or 5 (changing this smoothly is not yet supported)
        checkpoint (optional) = a state_dict from torch.load
        model, model_args: pass in a different model to use, and the arguments needed to instantiate it
        """
        super().__init__()

        self.normalization = normalization
        print("** using normalization strategy", normalization)

        if model:
            # Pass in your own model
            self.unet = model(**model_args)
        else:
            # TODO parameters should be loaded from configuration, not hardcoded
            model_args = {
                "n_input_channels": n_input_channels,
                "start_kernel": 64,
                "depth": 5,
                "act_fn_encoder": "relu",
                "act_fn_decoder": "relu",
                "act_fn_bottleneck": "relu",
                "up_mode": "transpose",
                "do_BN": True,
                "outer_skip": True,
                "outer_skip_BN": False,
                "bias_conv_layer": True,
            }

            # Load the UNet
            self.unet = UNet(**model_args)

        # Set learning rate
        self.lr = lr

        # Weight decay
        self.weight_decay = weight_decay

        # Possible to resume from checkpoint
        if checkpoint:
            self.unet.load_state_dict(
                {k[5:]: v for k, v in checkpoint["state_dict"].items()}
            )  # cut off the 'unet.' in state dict???

        # Define loss function
        self.loss = loss_fn()
        # TODO may try huber loss with a reasonable delta for this problem

    def forward(self, x):
        """Forward pass"""

        if self.normalization == "minmax":
            normalized_inputs = torch.zeros_like(x)

            min_max_per_band = [[0, 3266.77], [0, 2045], [0, 2007], [0, 6.98], [0, 1]]
            for band in range(len(min_max_per_band)):
                normalized_inputs[:, band] = (
                    x[:, band] - min_max_per_band[band][0]
                ) / min_max_per_band[band][1]

            out = self.unet(normalized_inputs)
            # scaling up the output from 0-1 to the range of DSM values
            out = (out * min_max_per_band[0][1]) + min_max_per_band[0][0]

        elif self.normalization == "meanstd":

            # Normalization of inputs needs to be flexible according to input configuration
            # TODO this method is going lead to bugs when adding many more types of layers
            # TODO refactor all normalization into utility functions
            # and handle different layer configurations
            dsm_mean = x[:, 0].mean()  # mean within batch, like Stucker & Schindler
            normalized_inputs = torch.zeros_like(x)
            normalized_inputs[:, 0] = torchvision.transforms.Normalize(
                dsm_mean, dsm_std
            )(x[:, 0])
            normalized_inputs[:, 1:3] = torchvision.transforms.Normalize(
                ortho_mean, ortho_std
            )(x[:, 1:3])
            # TODO normalization of trierror
            normalized_inputs[:, 3] = torchvision.transforms.Normalize(
                trierror_mean, trierror_std
            )(x[:, 3])
            # leave the mask alone, 0 or 1
            normalized_inputs[:, 4] = x[:, 4]

            # print(x[:,0].mean(), "mean of input DSM before normalization")
            out = self.unet(normalized_inputs)

            # Reverse DSM denormalization on the way out
            out = (out * dsm_std) + dsm_mean
            # print(out.mean(), "mean of output DSM from network after de-normalization")
        else:
            raise NotImplementedError
        return out

    def training_step(self, batch, batch_idx):
        """Take one training step and compute the loss"""
        output = self(batch["inputs"])
        # rescale the target ?
        train_loss = self.loss(output.squeeze(), batch["target"].squeeze())

        # Log all metrics
        # How to get these to show up aggregated at end of each epoch???
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )  # , on_epoch=True)

        # How to access tensorboard directly
        # tensorboard = self.logger.experiment
        # tensorboard.add_histogram()

        return train_loss

    def validation_step(self, batch, batch_idx):
        """Take one training step and compute the loss"""
        output = self(batch["inputs"])
        val_loss = self.loss(output.squeeze(), batch["target"].squeeze())

        # Log all metrics
        # How to get these to show up aggregated at end of each epoch???
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return val_loss

    def configure_optimizers(self):
        # Using Adam according to Stucker & Schindler
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
