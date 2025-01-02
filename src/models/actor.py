from .network_block_creator import create_network
from torch import nn
from entities.features import Run
import torch
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        sub_action_count = Run.instance().agent_config.sub_action_count
        config = {"final_activation":None , "activation":nn.ELU , "hidden_layer_count":1 , "shapes":[64]}
        self.networks = nn.ModuleList([create_network(config, input_shape=Run.instance().network_config.input_shape,output_shape=int(3),
                                      normalize_at_the_end=False, use_bias=True) for _ in range(sub_action_count)])
        
    def forward(self, x , module_index):
        output_mean = self.networks[module_index](x)
        mean = nn.Tanh()(output_mean)
        std = 0.4 * torch.ones_like(mean)
        return mean , std
    
    def act(self, x):
        sub_action_count = Run.instance().agent_config.sub_action_count
        means = [self.networks[i](x) for i in range(sub_action_count)]
        stds = [0.4 * torch.ones_like(means[i]) for i in range(sub_action_count)]
        return means , stds