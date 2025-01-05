from .network_block_creator import create_network
from torch import nn
from entities.features import Run


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        config = {
            "final_activation": None,
            "activation": nn.ELU,
            "hidden_layer_count": 2,
            "shapes": [128, 128]
        }
        self.network = create_network(config,
                                      input_shape=Run.instance().network_config.input_shape,
                                      output_shape=1,
                                      normalize_at_the_end=False,
                                      use_bias=True)

    def forward(self, x):
        output = self.network(x)
        # print(x.mean() , x.min() , x.max())
        return output
