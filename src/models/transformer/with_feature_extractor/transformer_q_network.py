from torch.nn import Module
from models.network_block_creator import create_network
from entities.features import Run
from torch import Tensor, cat
from .feature_extractor import FeatureExtractor


class TransformerQNetwork(Module):

    def __init__(self, environment_feature_extractor: FeatureExtractor):
        super(TransformerQNetwork, self).__init__()
        run = Run.instance()
        hidden_dim = run.network_config.feature_extractor_latent_size
        self.environment_feature_extractor = environment_feature_extractor
        self.feature_extractor = FeatureExtractor()
        self.activation_class = run.network_config.activation_class
        self.use_bias = run.network_config.use_bias
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        net_input_dim = int(hidden_dim + run.network_config.output_shape)
        self.first_network = create_network(config,
                                            net_input_dim,
                                            1,
                                            False,
                                            run.network_config.use_bias,
                                            False,
                                            last_layer_std=run.network_config.last_layer_std)
        self.second_network = create_network(config,
                                             net_input_dim,
                                             1,
                                             False,
                                             run.network_config.use_bias,
                                             False,
                                             last_layer_std=run.network_config.last_layer_std)
        pass

    def forward(self, state: Tensor,
                action: Tensor):  # x of shape (batch_size, sequence_length, 346)
        environment_features = self.environment_feature_extractor.extract_features(state)
        features = self.feature_extractor.extract_features(state)
        total_features = cat([features, environment_features], dim=1)
        input_tensor = cat([total_features, action], 1)
        out1, out2 = self.first_network(input_tensor), self.second_network(input_tensor)
        return out1, out2
