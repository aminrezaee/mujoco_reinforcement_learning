from torch.nn import Module, Sequential, Linear, TransformerEncoderLayer, TransformerEncoder
from torch import Tensor, cat
from entities.features import Run
from models.transformer.positional_encoding import SinusoidalPositionalEncoding
"""
    the goal of feature extractor is to regenerate the current observation from the current observation.
    after training the feature extractor, we use the extractor subnetwork to extract features from the observation
    """


class FeatureExtractor(Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
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
        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=run.network_config.num_feature_extractor_layers)
        self.fully_connected = Sequential(
            Linear(int(run.environment_config.window_length * hidden_dim),
                   int(run.environment_config.window_length * hidden_dim / 4),
                   bias=self.use_bias), self.activation_class(),
            Linear(int(run.environment_config.window_length * hidden_dim / 4),
                   int(run.environment_config.window_length * hidden_dim / 16),
                   bias=self.use_bias), self.activation_class())
        self.output_layer = Linear(int(run.environment_config.window_length * hidden_dim / 16),
                                   int(run.environment_config.window_length *
                                       run.network_config.input_shape),
                                   bias=self.use_bias)

    def forward(self, state: Tensor):
        x = self.extract_features(state)
        x = self.output_layer(x)
        x = x.reshape((len(state), -1, Run.instance().network_config.input_shape))
        return x

    def extract_features(self, state: Tensor):
        x = self.positional_encoding(state)
        x = self.projection(x)
        x = self.encoder(x)
        x = x.reshape((len(state), -1))
        x = self.fully_connected(x)
        return x
