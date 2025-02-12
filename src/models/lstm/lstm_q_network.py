from models.network_block_creator import create_network
from torch import nn
from entities.features import Run
from torch import Tensor, cat


class LSTMQNetwork(nn.Module):

    def __init__(self):
        super(LSTMQNetwork, self).__init__()
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }

        self.feature_extractor = nn.Sequential(
            nn.LSTM(run.network_config.input_shape,
                    run.network_config.feature_extractor_latent_size,
                    bidirectional=True,
                    batch_first=True))
        fully_connected_input_shape = int(run.network_config.feature_extractor_latent_size * 2 +
                                          run.network_config.output_shape)
        self.first_network = create_network(
            config,
            input_shape=fully_connected_input_shape,  # due to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=run.network_config.use_bias,
            last_layer_std=run.network_config.last_layer_std)
        self.second_network = create_network(
            config,
            input_shape=fully_connected_input_shape,  # due to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=run.network_config.use_bias,
            last_layer_std=run.network_config.last_layer_std)

    def forward(self, state: Tensor, action: Tensor):
        # input_tensor = cat([state, action], 1)
        features = self.feature_extractor(state)
        current_timestep_features = features[0][:, -1, :]
        # current_timestep_features = features[0].reshape(len(features), -1)
        current_timestep_features = Run.instance().network_config.activation_class()(
            current_timestep_features)
        input_tensor = cat([current_timestep_features, action], 1)

        out1, out2 = self.first_network(input_tensor), self.second_network(input_tensor)
        # print(x.mean() , x.min() , x.max())
        return out1, out2
