from torch import nn
from .network_block_creator import create_network
from entities.features import Run
import torch


class LSTMActor(nn.Module):

    def __init__(self):
        super(LSTMActor, self).__init__()
        self.feature_extractor = nn.LSTM(376, 10, dropout=0.1, bidirectional=True)
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": 2,
            "shapes": [128, 128]
        }
        self.actor = create_network(config, int(2 * run.network_config.input_shape),
                                    run.network_config.output_shape, False,
                                    run.network_config.use_bias, False)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):
        run = Run.instance()
        features = self.feature_extractor(x)
        current_timestep_features = features[
            0][:, :, [0, Run.instance().environment_config.window_length]]
        current_timestep_features = current_timestep_features.view(
            current_timestep_features.size(0), -1)
        output = self.actor(current_timestep_features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
