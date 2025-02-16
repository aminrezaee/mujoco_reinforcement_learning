from torch.nn import Tanh, Parameter, Module
from models.network_block_creator import create_network
from entities.features import Run
import torch
from torch import cat
from .feature_extractor import FeatureExtractor


class TransformerActor(Module):

    def __init__(self, environment_feature_extractor: FeatureExtractor):
        super(TransformerActor, self).__init__()
        run = Run.instance()
        self.environment_feature_extractor = environment_feature_extractor
        self.feature_extractor = FeatureExtractor()
        hidden_dim = run.network_config.feature_extractor_latent_size
        config = {
            "final_activation": Tanh,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        self.actor = create_network(config,
                                    hidden_dim,
                                    run.network_config.output_shape,
                                    False,
                                    run.network_config.use_bias,
                                    False,
                                    last_layer_std=run.network_config.last_layer_std)
        self.actor_logstd = Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 346)
        run = Run.instance()
        environment_features = self.environment_feature_extractor.extract_features(x)
        features = self.feature_extractor.extract_features(x)
        total_features = cat([features, environment_features], dim=1)
        output = self.actor(total_features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
