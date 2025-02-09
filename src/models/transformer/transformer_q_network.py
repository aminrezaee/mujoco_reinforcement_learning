from torch.nn import Linear, Sequential, TransformerEncoderLayer, TransformerEncoder, Transformer, Tanh, Parameter, Module
from models.network_block_creator import create_network
from entities.features import Run
import torch
from torch import Tensor, cat


class TransformerQNetwork(Module):

    def __init__(self):
        super(TransformerQNetwork, self).__init__()
        run = Run.instance()
        input_dim = run.network_config.input_shape
        hidden_dim = run.network_config.feature_extractor_latent_size
        self.activation_class = run.network_config.activation_class
        self.use_bias = run.network_config.use_bias
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
        projections = self.projection(state)
        features = self.encoder(projections).reshape(
            len(state), -1)  # features of shape (batch_size, sequence_length, hidden_dim)
        compressed_features = self.compression(features)
        features = Run.instance().network_config.activation_class()(compressed_features)
        input_tensor = cat([features, action], 1)
        out1, out2 = self.first_network(input_tensor), self.second_network(input_tensor)
        return out1, out2
