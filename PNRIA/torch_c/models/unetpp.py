import torch
from torch import nn

from PNRIA.torch_c.custom_model import BaseModel
from deep_filaments.torch.third_party.layers import unetConv2, unetUp_origin
from deep_filaments.torch.third_party.init_weights import init_weights


class UNetPP(BaseModel):
    """
    UNetPlusPlus model
    """

    required_keys = ["in_channels", "n_classes", "feature_scale", "is_deconv", "is_batchnorm", "is_ds", "n_blocks", "filters"]
    aliases = ["UNetPP"]

    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
            is_ds=True,
            n_blocks=[2, 2, 2, 2],  # number of blocks in each column
            filters=[64, 128, 256, 512, 1024],  # number of filters in each block
            **kwargs  # Capture additional arguments for configuration
    ):
        super(UNetPP, self).__init__()

        self._init_layer()

    def _init_layer(self):
        # downsampling
        self.down_layers = nn.ModuleList()
        in_size = self.in_channels
        for i, n_block in enumerate(self.n_blocks):
            layers = []
            for j in range(n_block):
                layers.append(unetConv2(in_size, self.filters[i], self.is_batchnorm))
                in_size = self.filters[i]
            if i < len(self.n_blocks) - 1:
                layers.append(torch.nn.MaxPool2d(kernel_size=2))
            self.down_layers.append(nn.Sequential(*layers))
        # upsampling
        self.up_layers = nn.ModuleList()
        for i in range(len(self.n_blocks) - 1, 0, -1):
            layers = []
            for j in range(self.n_blocks[i]):
                in_size = self.filters[i] * (j + 2)
                out_size = self.filters[i - 1]
                layers.append(unetUp_origin(in_size, out_size, self.is_deconv, j + 2))
            self.up_layers.append(nn.ModuleList(layers))
        # final conv (without any concat)
        self.final_layers = nn.ModuleList(
            [torch.nn.Conv2d(self.filters[0], self.n_classes, (1, 1)) for _ in range(len(self.n_blocks))])
        # initialise weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, torch.nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def _preprocess_forward(self, inputs):
        """Prétraitement avant l'opération de forward"""
        return inputs  # Ici, il n'y a pas de prétraitement spécifique, on passe simplement les entrées.

    def _core_forward(self, inputs):
        """Opération principale de forward"""
        # downsampling
        x = inputs
        down_outputs = []
        for layer in self.down_layers:
            x = layer(x)
            down_outputs.append(x)

        # upsampling
        up_outputs = []
        for i, layer_group in enumerate(self.up_layers):
            x = down_outputs[-1]
            for j, layer in enumerate(layer_group):
                x = layer(x, *down_outputs[-(j+2):])
            up_outputs.append(x)
            down_outputs.pop()

        # final layer
        final_outputs = [final_layer(up_output) for final_layer, up_output in zip(self.final_layers, up_outputs)]
        final = torch.mean(torch.stack(final_outputs), dim=0)

        if self.is_ds:
            return torch.sigmoid(final)
        else:
            return torch.sigmoid(final_outputs[-1])



# region layer factory methods

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find("unetConv2") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find("unetConv2") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

# endregion