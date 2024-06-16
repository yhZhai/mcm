import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange


class SpatialHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        assert num_layers >= 2, "num_layers must be at least 2"

        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.in_act = nn.SiLU()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        num_channels if i == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                    ),
                    nn.GroupNorm(
                        num_groups=norm_num_groups,
                        num_channels=hidden_dim,
                        eps=norm_eps,
                    ),
                    nn.SiLU(),
                )
                for i in range(num_layers - 1)
            ]
        )

        self.conv_out = nn.Conv2d(hidden_dim, num_channels, kernel_size=1, padding=0)

        # zero initialize the last layer
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        # x shape: b, c, t, h, w
        b, c, t, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.in_act(x)
        for layer in self.layers:
            x = layer(x)

        x = self.conv_out(x)

        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        x = x + x_in
        return x


class IdentitySpatialHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self):
        super().__init__()

        self.identity_layer = nn.Identity()

    def forward(self, x):
        return self.identity_layer(x)


if __name__ == "__main__":
    head = SpatialHead(num_channels=4, num_layers=3)
    x = torch.randn(2, 4, 3, 32, 32)
    out = head(x)
    diff = out - x
    print(diff.abs().max())
