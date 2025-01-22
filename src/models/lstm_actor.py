from torch import nn
from .network_block_creator import create_network
from entities.features import Run
import torch
from memory_profiler import profile


class LSTMActor(nn.Module):

    def __init__(self):
        super(LSTMActor, self).__init__()
        self.feature_extractor = nn.LSTM(376,
                                         Run.instance().network_config.latent_size,
                                         num_layers=1,
                                         bidirectional=True,
                                         batch_first=True)
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": 2,
            "shapes": [128, 64]
        }
        self.actor = create_network(
            config, int(run.network_config.latent_size * 2 + run.network_config.input_shape),
            run.network_config.output_shape, False, run.network_config.use_bias, False)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 376)
        run = Run.instance()
        features = self.feature_extractor(x)[
            0]  # features of shape (batch_size, sequence_length, 20)
        features = Run.instance().network_config.activation_class()(features)
        current_timestep_features = features[:, -1, :]
        last_timestep_features = x[:, -1]
        # current_timestep_features = features[0].reshape(len(features), -1)
        input_tensor = torch.cat([current_timestep_features, last_timestep_features], dim=1)
        output = self.actor(input_tensor)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
