from .network_block_creator import create_network
from torch import nn
from entities.features import Run
import torch


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        run = Run.instance()
        sub_action_count = run.agent_config.sub_action_count
        config = {
            "final_activation": None,
            "activation": nn.Tanh,
            "hidden_layer_count": 2,
            "shapes": [128, 128]
        }
        self.networks = nn.ModuleList([
            create_network(config,
                           input_shape=run.network_config.input_shape,
                           output_shape=int(run.network_config.output_shape / sub_action_count),
                           normalize_at_the_end=False,
                           use_bias=True) for _ in range(sub_action_count)
        ])
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))

    def forward(self, x, module_index):
        run = Run.instance()
        output_mean = self.networks[module_index](x)
        mean = run.network_config.output_max_value * nn.Tanh()(output_mean)
        sub_action_count = run.agent_config.sub_action_count
        sub_action_size = int(run.network_config.output_shape / sub_action_count)
        std = self.actor_logstd[int(module_index * sub_action_size):int((module_index + 1) *
                                                                        sub_action_size)].exp()
        return mean, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x):
        run = Run.instance()
        sub_action_count = run.agent_config.sub_action_count
        sub_action_size = int(run.network_config.output_shape / sub_action_count)
        means = [self.networks[i](x)[None, :] for i in range(sub_action_count)]
        # print(means[0].mean() , means[0].min() , means[0].max())
        stds = [
            self.actor_logstd[int(i * sub_action_size):int((i + 1) *
                                                           sub_action_size)].exp()[None, None, :]
            for i in range(sub_action_count)
        ]
        return torch.cat(means), torch.cat(stds)
