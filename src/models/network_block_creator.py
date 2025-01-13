import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Dropout, Linear, Module, Sequential, Tanh, GELU, ELU
import numpy as np


class InputNormalization(Module):

    def forward(self, x_in: Tensor):
        x_in = x_in - x_in.mean(dim=-1)[..., None]
        std = x_in.std(dim=-1)[..., None]
        std = torch.where(std == 0, torch.tensor(1e-4).to(std.device), std)
        x_in = x_in / std
        return x_in


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NetworkBlock(Module):

    def __init__(self,
                 config: dict,
                 input_shape,
                 output_shape,
                 normalize_at_the_end: bool,
                 use_bias: bool,
                 use_batchnorm=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_bias = use_bias
        self.hidden_layer_count = config["hidden_layer_count"]
        if 'skip_connection' not in list(config.keys()):
            self.use_skip_connections: bool = False
        else:
            self.use_skip_connections: bool = config['skip_connection']
        self.normalize_at_the_end = normalize_at_the_end
        layers = []
        for i in range(self.hidden_layer_count):
            out_shape = config["shapes"][i]
            layer = Linear(input_shape, out_shape, bias=self.use_bias)
            with torch.no_grad():
                layer = layer_init(layer)
                # if self.use_bias:
                #     layer.bias.fill_(0)
            layers.append(layer)
            if config["activation"] is not None:
                layers.append(config["activation"]())
                # layers.append(Dropout(0.1))
            # layers.append(InputNormalization())
            input_shape = out_shape
        # layers.append(Dropout(0.2))
        # if use_batchnorm:
        #     layers.append(BatchNorm1d(out_shape))
        self.first_layers = Sequential(*layers)
        self.last_layer = Linear(out_shape, output_shape, bias=self.use_bias)
        with torch.no_grad():
            self.last_layer = layer_init(self.last_layer, std=0.01)
            # if self.use_bias:
            #     self.last_layer.bias.fill_(0)
        if config["final_activation"] is not None:
            if config["final_activation"] in [Tanh, GELU, ELU]:
                self.last_layer_activation = config["final_activation"]()
            else:
                self.last_layer_activation = config["final_activation"](dim=-1)
        else:
            self.last_layer_activation = None

    def forward(self, x_in: Tensor):
        x_out = self.first_layers(x_in)
        x_out = self.last_layer(x_out)
        # print(x)
        if self.last_layer_activation is not None:
            x_out = self.last_layer_activation(x_out)
        # normalize results
        if self.normalize_at_the_end:
            x_out = InputNormalization()(x_out)
        if self.use_skip_connections:
            return torch.cat([x_in, x_out], dim=1)
        # print(x)
        return x_out


def create_network(config,
                   input_shape,
                   output_shape,
                   normalize_at_the_end: bool,
                   use_bias,
                   use_batchnorm=False):
    return NetworkBlock(config,
                        input_shape,
                        output_shape,
                        normalize_at_the_end=normalize_at_the_end,
                        use_bias=use_bias,
                        use_batchnorm=use_batchnorm)
