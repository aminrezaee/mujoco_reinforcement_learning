from torch import nn
from .network_block_creator import create_network
from entities.features import Run
import torch


class LSTMActor(nn.Module):

    def __init__(self):
        super(LSTMActor, self).__init__()
        self.feature_extractor = nn.LSTM(376, 10, dropout=0.1, bidirectional=True, batch_first=True)
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": 2,
            "shapes": [128, 128]
        }
        self.actor = create_network(config, 20, run.network_config.output_shape, False,
                                    run.network_config.use_bias, False)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 376)
        run = Run.instance()
        features = self.feature_extractor(x)  # features of shape (batch_size, sequence_length, 20)
        current_timestep_features = features[0][:, -1, :]
        current_timestep_features = Run.instance().network_config.activation_class()(
            current_timestep_features)
        output = self.actor(current_timestep_features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
