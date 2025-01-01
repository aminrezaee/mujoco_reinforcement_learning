from .network_block_creator import create_network
from torch import nn
from entities.features import Run
import torch
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        config = {"final_activation":None , "activation":nn.ELU , "hidden_layer_count":1 , "shapes":[64]}
        self.network = create_network(config, input_shape=Run.instance().network_config.input_shape,output_shape=int(21),
                                      normalize_at_the_end=False, use_bias=True)
        
    def forward(self, x):
        x_std = torch.repeat_interleave(x, 20 , dim=0)
        x_std = x_std + 0.01 * torch.randn_like(x_std)
        output_mean = self.network(x)
        means = nn.Tanh()(output_mean)
        output_std:torch.Tensor = self.network(x_std)
        stds = output_std.std(dim=0)
        return means , stds