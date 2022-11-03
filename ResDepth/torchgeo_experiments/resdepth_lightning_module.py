import sys

import torch
import torchvision
from pytorch_lightning import LightningModule

# TODO figure out how to change to relative import of ResDepth UNet
sys.path.insert(0, "/mnt/1.0_TB_VOLUME/sethv/resdepth_all/ResDepth/")
from lib.UNet import UNet

from train_utils import dsm_std, ortho_mean, ortho_std


class TGDSMLightningModule(LightningModule):
    """Module to train ResDepth with PyTorch Lightning"""

    def __init__(self, n_input_channels):
        super().__init__()

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

        # Possible to resume from checkpoint
        # self.unet.load_state_dict(resdepth_state_dict["model_state_dict"])

        # Define loss function
        self.loss = torch.nn.L1Loss()

    def forward(self, x):
        """Forward pass"""

        # Normalization of inputs needs to be flexible according to input configuration
        # TODO this is going to be gross when adding many more types of layers
        dsm_mean = x[:, 0].mean()  # mean within batch, like Stucker & Schindler
        normalized_inputs = torch.zeros_like(x)
        normalized_inputs[:, 0] = torchvision.transforms.Normalize(dsm_mean, dsm_std)(
            x[:, 0]
        )
        normalized_inputs[:, 1:3] = torchvision.transforms.Normalize(
            ortho_mean, ortho_std
        )(x[:, 1:3])

        # print(x[:,0].mean(), "mean of input DSM before normalization")
        # print(normalized_inputs[:,0].mean(), "mean of input DSM after normalization")

        if x.shape[1] > 3:
            # TODO handle all layers properly
            # Just pass through inputs, we don't know how to normalize them
            normalized_inputs[:, 3:] = x[:, 3:]
        # plot_batch(normalized_inputs)  # debug
        out = self.unet(normalized_inputs)
        # print(out.mean(), "mean of output DSM from network before de-normalization")

        # Reverse DSM denormalization on the way out
        out = (out * dsm_std) + dsm_mean
        # print(out.mean(), "mean of output DSM from network after de-normalization")
        return out

    def training_step(self, batch, batch_idx):
        """Take one training step and compute the loss"""
        output = self(batch["inputs"])
        train_loss = self.loss(output.squeeze(), batch["target"].squeeze())

        log_dict = {"loss": train_loss, "train_loss": train_loss}

        # Log all metrics
        # How to get these to show up aggregated at end of each epoch???
        self.log("loss", train_loss)  # , on_epoch=True)
        self.log_dict(log_dict)  # , on_epoch=True)
        # self.logger.log_metrics(log_dict)

        # How to access tensorboard directly
        # tensorboard = self.logger.experiment
        # tensorboard.add_histogram()

        return train_loss

    def configure_optimizers(self):
        # Using Adam according to Stucker & Schindler
        # TODO no reason to hard code these parameters
        return torch.optim.Adam(self.parameters(), lr=0.00002, weight_decay=1e-5)
