from x_unet import XUnet


class XUnetWithSkipConnection(XUnet):
    """Simple modification of fancy [X-UNet](https://github.com/lucidrains/x-unet/)

    Add Stucker et al residual connection from UNet initial input DSM to the output,
    so that network learns to compute the residual correction like ResDepth."""

    def forward(self, x):
        residual = super().forward(x)
        x_0 = x[:, 0, :, :]  # initial DSM passed in to the network
        x_0 = x_0.unsqueeze(1)

        return x_0 + residual
