from torch import nn
from models.network_block_creator import create_network
from entities.features import Run
import torch


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        self.actor = create_network(
            config, int(run.network_config.input_shape * run.environment_config.window_length),
            run.network_config.output_shape, False, run.network_config.use_bias,
            run.network_config.use_batch_norm)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 346)
        x = x.reshape(len(x), -1)
        run = Run.instance()
        output = self.actor(x)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
