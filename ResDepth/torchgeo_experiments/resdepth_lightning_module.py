import sys
import importlib

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
        n_input_channels=5,
        lr=0.00002,
        weight_decay=1e-5,
        checkpoint=None,
        loss_fn_module_name="torch.nn",
        loss_fn_class_name="L1Loss",
        # loss_fn=torch.nn.L1Loss,
        model="ResDepth_UNet",
        model_args=None,
        normalization="meanstd",
        metrics=[],
        use_input_dem_mask_for_computing_loss=False
    ):
        """
        n_input_channels 3 or 4 or 5 (changing this smoothly is not yet supported)
        checkpoint (optional) = a state_dict from torch.load
        model, model_args: pass in a different model to use, and the arguments needed to instantiate it
        """
        super().__init__()

        # Log each of the metrics for train & validation datasets
        self.metrics = metrics

        self.normalization = normalization
        print("** using normalization strategy", normalization)

        self.use_input_dem_mask_for_computing_loss = use_input_dem_mask_for_computing_loss

        if model and model != "ResDepth_UNet":
            # Pass in your own model
            self.unet = model(**model_args)
        else:
            # Assume we are using standard ResDepth UNet
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
        # self.loss = loss_fn  # old way didn't work with LightningCLI
        module = importlib.import_module(loss_fn_module_name)
        class_ = getattr(module, loss_fn_class_name)
        instance = class_()
        self.loss = instance
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

            # Forward pass of model
            out = self.unet(normalized_inputs)

            # Reverse DSM denormalization on the way out
            out = (out * dsm_std) + dsm_mean
            # print(out.mean(), "mean of output DSM from network after de-normalization")
        else:
            raise NotImplementedError

        return out

    def training_step(self, batch, batch_idx):
        """Take one training step and compute the loss"""
        # ins = batch["inputs"]
        output = self(batch["inputs"]).squeeze()
        target = batch["target"].squeeze()

        # nan_mask = torch.zeros_like(target)
        # print(f"Loss before replacement= {self.loss(output, target)}")
        # print(torch.isfinite(output).sum())
        # print(torch.isfinite(target).sum())
        output = torch.where(torch.isfinite(output),output,0)
        output = torch.where(torch.isfinite(target),output,0)
        target = torch.where(torch.isfinite(output),target,0)
        target = torch.where(torch.isfinite(target),target,0)
        # print(torch.isfinite(output).sum())
        # print(torch.isfinite(target).sum())
        # print(f"Sum of output= {output.sum()}")
        # print(f"Sum of target= {target.sum()}")
        # print(f"Loss after replacement= {self.loss(output, target)}")
        # input("Pasued here")

        # target=torch.nan_to_num(target,0)
        # torch.bitwise_or(output < 0, output > 5000, target < 0, target > 5000, torch.isfinite(output), torch.isfinite(target), nan_mask)
        # output = torch.where(nan_mask, output, 0)
        # target = torch.where(nan_mask, target, 0)

        # print(nan_mask.shape, nan_mask.sum())
        # for layer in range(ins.shape[1]):
        #     print(f"Number of inputs > 0: {torch.sum(ins[:,layer] > 0)} shape= {ins[:,layer].shape}")

        # print(f"Number of outputs > 0: {torch.sum(output > 0)} shape= {output.shape}")
        # print(f"Number of target > 0: {torch.sum(target > 0)} shape= {target.shape}")
        # # Mask out all NaNs of output & target first ???

        # print(f"Sum of nan_mask joint output & target {torch.sum(nan_mask)}")

        # input("INput here to keep going:")


        train_loss_before = self.loss(output, target)
        train_loss = train_loss_before

        # TODO move this to config file!!!
        if self.use_input_dem_mask_for_computing_loss:

            # TODO fix the operation of the loss function
            # See what happens if we mask all pixels where input DEM had nodata
            # We are using an inpainted DEM as the actual input, but
            # the NN receives the stereo nodata mask as input:
            # in theory it could learn to fix these holes,
            # but we want to see what happens if we avoid training to fix them
            # and only compute loss where input DEM was valid
            nodata_mask_layer_idx = -1 # yuck, assumes nodata is last
            mask = batch["inputs"][:,nodata_mask_layer_idx]

            # rescale the target ?
            num_valid_output_px_before = (output > 0).sum(dim=[1,2]).cpu().numpy()
            # print(f"Output before: number of pixels={num_valid_output_px_before}")


            debug = False
            if debug:
                print("Masking")
                print(f"Loss before: {train_loss_before:.2f}")
            output = torch.where(mask > 0, output, 0)
            target = torch.where(mask > 0, target, 0)
            num_valid_output_px_after = (output > 0).sum(dim=[1,2]).cpu().numpy()
            diff = num_valid_output_px_after - num_valid_output_px_before

            # print(f"Output after: number of pixels={num_valid_output_px_after}")


            # TODO want to be able to pass inputs to loss function rather than
            train_loss = self.loss(output, target)

            if debug:
                print(f"Removed {-1 * diff.sum()} pixels in batch, max in 1 patch was {-1 * diff.min()}")
                print(f"Loss after: {train_loss:.2f}")
                input("Hit enter to go to next: ")

        # IF
        if not torch.isfinite(train_loss):
            train_loss = torch.zeros_like(train_loss)

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

    def on_training_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics:
            # compute and log metric for this step
            metric(outputs["preds"], outputs["target"])
            self.log(f"train_{metric_name}", metric)


        return super().training_epoch_end(outputs)


    def validation_step(self, batch, batch_idx):
        """Take one training step and compute the loss"""
        output = self(batch["inputs"]).squeeze()

        target = batch["target"].squeeze()
        val_loss = self.loss(output, target)
        l1_loss = torch.nn.L1Loss()(output, target)
        l2_loss = torch.nn.MSELoss()(output, target)
        val_metrics = dict(val_loss=val_loss, val_l1_loss=l1_loss, val_l2_loss=l2_loss)


        nodata_mask_layer_idx = -1
        mask = batch["inputs"][:,nodata_mask_layer_idx]

        masked_output = torch.where(mask > 0, output, 0)
        masked_target = torch.where(mask > 0, target, 0)
        masked_val_loss = self.loss(masked_output, masked_target)
        masked_l1_loss = torch.nn.L1Loss()(masked_output, masked_target)
        masked_l2_loss = torch.nn.MSELoss()(masked_output, masked_target)


        masked_val_metrics = dict(masked_val_loss=masked_val_loss, masked_val_l1_loss=masked_l1_loss, masked_val_l2_loss=masked_l2_loss)

        val_metrics.update(masked_val_metrics)

        # TODO validation metrics with mask
        # val_metrics_masked
        self.log_dict(
            val_metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        return val_loss

    def configure_optimizers(self):
        # Using Adam according to Stucker & Schindler
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
