from torch.nn import Tanh, Parameter, Module
from models.network_block_creator import create_network
from entities.features import Run
import torch
from .feature_extractor import FeatureExtractor


class TransformerActor(Module):

    def __init__(self, feature_extractor: FeatureExtractor):
        super(TransformerActor, self).__init__()
        run = Run.instance()
        self.feature_extractor = feature_extractor
        hidden_dim = run.network_config.feature_extractor_latent_size
        config = {
            "final_activation": Tanh,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }
        self.actor = create_network(config,
                                    int(hidden_dim / 2),
                                    run.network_config.output_shape,
                                    False,
                                    run.network_config.use_bias,
                                    False,
                                    last_layer_std=run.network_config.last_layer_std)
        self.actor_logstd = Parameter(torch.zeros(run.network_config.output_shape))
        pass

    def forward(self, x):  # x of shape (batch_size, sequence_length, 346)
        run = Run.instance()
        features = self.feature_extractor.extractor(x)
        output = self.actor(features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
