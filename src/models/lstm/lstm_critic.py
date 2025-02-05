from models.network_block_creator import create_network
from torch import nn
from torch import Tensor
from entities.features import Run


class LSTMCritic(nn.Module):

    def __init__(self):
        super(LSTMCritic, self).__init__()
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }

        self.feature_extractor = nn.Sequential(
            nn.LSTM(run.network_config.input_shape,
                    run.network_config.lstm_latent_size,
                    bidirectional=True,
                    batch_first=True))
        fully_connected_input_shape = int(run.network_config.lstm_latent_size * 2)
        self.network = create_network(
            config,
            input_shape=fully_connected_input_shape,  # due to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=run.network_config.use_bias)

    def forward(self, x: Tensor):
        features = self.feature_extractor(x)
        current_timestep_features = features[0][:, -1, :]
        # current_timestep_features = features[0][:, -1, :]
        current_timestep_features = Run.instance().network_config.activation_class()(
            current_timestep_features)
        output = self.network(current_timestep_features)
        # print(x.mean() , x.min() , x.max())
        return output
