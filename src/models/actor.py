from .network_block_creator import create_network
from torch import nn
from entities.features import Run
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        config = {"final_activation":None , "activation":nn.ELU , "hidden_layer_count":1 , "shapes":[64]}
        self.network = create_network(config, input_shape=Run.instance().network_config.input_shape,output_shape=int(2*21),
                                      normalize_at_the_end=False, use_bias=True)
        
    def forward(self, x):
        output = self.network(x)
        means = nn.Tanh()(output[:,:21])
        stds = nn.Sigmoid()(output[:,21:])
        return means , stds