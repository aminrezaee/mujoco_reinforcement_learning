from .network_block_creator import create_network
from torch import nn
from entities.features import Run
from torch import Tensor, cat


class LSTMQNetwork(nn.Module):

    def __init__(self):
        super(LSTMQNetwork, self).__init__()
        config = {
            "final_activation": None,
            "activation": Run.instance().network_config.activation_class,
            "hidden_layer_count": 2,
            "shapes": [128, 64]
        }

        self.feature_extractor = nn.Sequential(
            nn.LSTM(376,
                    Run.instance().network_config.latent_size,
                    bidirectional=True,
                    batch_first=True))
        self.first_network = create_network(
            config,
            input_shape=int(Run.instance().network_config.latent_size * 2 *
                            Run.instance().environment_config.window_length
                            ),  # duo to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=True)
        self.second_network = create_network(
            config,
            input_shape=int(Run.instance().network_config.latent_size * 2 *
                            Run.instance().environment_config.window_length
                            ),  # duo to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=True)

    def forward(self, state: Tensor, action: Tensor):
        input_tensor = cat([state, action], 1)
        features = self.feature_extractor(input_tensor)
        current_timestep_features = features[0].reshape(len(features), -1)
        # current_timestep_features = features[0][:, -1, :]
        current_timestep_features = Run.instance().network_config.activation_class()(
            current_timestep_features)
        out1, out2 = self.first_network(current_timestep_features), self.second_network(
            current_timestep_features)
        # print(x.mean() , x.min() , x.max())
        return out1, out2
