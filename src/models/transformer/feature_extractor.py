from torch.nn import Module, Sequential, Linear, TransformerEncoderLayer, TransformerEncoder
from entities.features import Run
"""
    the goal of feature extractor is to predict the next timestep observation from the current observation.
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
        projection = Sequential(Linear(input_dim, hidden_dim, bias=self.use_bias),
                                self.activation_class())
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim,
                                                nhead=8,
                                                dim_feedforward=int(2 * hidden_dim),
                                                dropout=0.1,
                                                activation=self.activation_class(),
                                                bias=self.use_bias,
                                                batch_first=True)
        encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                     num_layers=run.network_config.num_feature_extractor_layers)
        fully_connected = Sequential(
            Linear(int(hidden_dim * run.environment_config.window_length),
                   int(hidden_dim / 2),
                   bias=self.use_bias), self.activation_class())
        self.extractor = Sequential(projection, encoder, fully_connected)
        self.output_layer = Sequential(
            Linear(int(hidden_dim / 2), hidden_dim, bias=self.use_bias), self.activation_class(),
            Linear(hidden_dim, run.network_config.input_shape, bias=self.use_bias))

    def forward(self, x):
        x = self.extractor(x)
        return self.output_layer(x)
