from torch.nn import Linear, Sequential, TransformerEncoderLayer, TransformerEncoder, Tanh, Parameter, Module
from models.network_block_creator import create_network
from entities.features import Run
import torch
from .positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class TransformerActor(Module):

    def __init__(self):
        super(TransformerActor, self).__init__()
        run = Run.instance()
        input_dim = run.network_config.input_shape
        hidden_dim = run.network_config.feature_extractor_latent_size
        self.activation_class = run.network_config.activation_class
        self.use_bias = run.network_config.use_bias
        self.positional_encoding = SinusoidalPositionalEncoding()
        self.projection = Sequential(Linear(input_dim, hidden_dim, bias=self.use_bias),
                                     self.activation_class())
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_dim,
                                                     nhead=8,
                                                     dim_feedforward=int(2 * hidden_dim),
                                                     dropout=0.1,
                                                     activation=self.activation_class(),
                                                     bias=self.use_bias,
                                                     batch_first=True)

        self.compression = Sequential(
            Linear(int(hidden_dim * run.environment_config.window_length),
                   hidden_dim,
                   bias=self.use_bias), self.activation_class())
        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=run.network_config.num_feature_extractor_layers)
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
        x = self.positional_encoding(x)
        projections = self.projection(x)
        features = self.encoder(projections).reshape(
            len(x), -1)  # features of shape (batch_size, sequence_length, 20)
        compressed_features = self.compression(features)
        features = Run.instance().network_config.activation_class()(compressed_features)
        output = self.actor(features)
        std = self.actor_logstd[:run.network_config.output_shape].exp()
        return output, torch.repeat_interleave(std[None, :], x.shape[0], dim=0)

    def act(self, x: torch.Tensor):
        return self.forward(x)
