from torch import nn
from .network_block_creator import create_network
from entities.features import Run
import torch
from memory_profiler import profile


class LSTMActor(nn.Module):

    def __init__(self):
        super(LSTMActor, self).__init__()
        run = Run.instance()
        self.feature_extractor = nn.LSTM(run.network_config.input_shape,
                                         run.network_config.lstm_latent_size,
                                         num_layers=run.network_config.num_lstm_layers,
                                         bidirectional=True,
                                         batch_first=True)
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        self.actor = create_network(
            config,
            int(Run.instance().network_config.lstm_latent_size * 2 *
                Run.instance().environment_config.window_length), run.network_config.output_shape,
            False, run.network_config.use_bias, False)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 376)
        run = Run.instance()
        features = self.feature_extractor(x)[0].reshape(
            len(x), -1)  # features of shape (batch_size, sequence_length, 20)
        features = Run.instance().network_config.activation_class()(features)
        output = self.actor(features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
