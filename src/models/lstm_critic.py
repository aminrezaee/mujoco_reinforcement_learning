from .network_block_creator import create_network
from torch import nn
from entities.features import Run


class LSTMCritic(nn.Module):

    def __init__(self):
        super(LSTMCritic, self).__init__()
        config = {
            "final_activation": None,
            "activation": Run.instance().network_config.activation_class,
            "hidden_layer_count": 2,
            "shapes": [128, 128]
        }
        self.feature_extractor = nn.LSTM(376, 10, dropout=0.1, bidirectional=True, batch_first=True)
        self.network = create_network(
            config,
            input_shape=20,  # duo to bidirectional feature extractor
            output_shape=1,
            normalize_at_the_end=False,
            use_bias=True)

    def forward(self, x):
        features = self.feature_extractor(x)
        current_timestep_features = features[0][:, -1, :]
        output = self.network(current_timestep_features)
        # print(x.mean() , x.min() , x.max())
        return output
