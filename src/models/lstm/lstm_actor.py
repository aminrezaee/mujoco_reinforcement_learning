from torch import nn
from models.network_block_creator import create_network
from entities.features import Run
import torch


class LSTMActor(nn.Module):

    def __init__(self):
        super(LSTMActor, self).__init__()
        run = Run.instance()
        self.feature_extractor = nn.LSTM(run.network_config.input_shape,
                                         run.network_config.feature_extractor_latent_size,
                                         num_layers=run.network_config.num_feature_extractor_layers,
                                         bidirectional=True,
                                         batch_first=True)
        config = {
            "final_activation": nn.Tanh,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        self.actor = create_network(config,
                                    int(run.network_config.feature_extractor_latent_size * 2 *
                                        run.environment_config.window_length),
                                    run.network_config.output_shape,
                                    False,
                                    run.network_config.use_bias,
                                    False,
                                    last_layer_std=run.network_config.last_layer_std)
        self.actor_logstd = nn.Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 346)
        run = Run.instance()
        features = self.feature_extractor(x)[0].reshape(
            len(x), -1)  # features of shape (batch_size, sequence_length, 20)
        features = Run.instance().network_config.activation_class()(features)
        output = self.actor(features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
