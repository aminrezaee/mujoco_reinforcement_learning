from entities.features import Run

from torch.nn import Module, Embedding
import torch
import math


class LearnedPositionalEncoding(Module):

    def __init__(self):
        super(LearnedPositionalEncoding, self).__init__()
        run = Run.instance()
        self.pos_embedding = Embedding(run.environment_config.window_length,
                                       run.network_config.input_shape)

    def forward(self, x):
        """Add positional encoding to the input tensor x."""
        # x is expected to have shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
        position_embeddings = self.pos_embedding(positions)  # Shape: (1, seq_len, d_model)
        return x + position_embeddings


class SinusoidalPositionalEncoding(Module):

    def __init__(self):
        super(SinusoidalPositionalEncoding, self).__init__()
        run = Run.instance()
        self.d_model = run.network_config.input_shape

        # Compute the positional encodings once in log space
        pe = torch.zeros(run.environment_config.window_length, run.network_config.input_shape)
        position = torch.arange(0, run.environment_config.window_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, run.network_config.input_shape, 2).float() *
            (-math.log(10000.0) / run.network_config.input_shape))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input tensor x."""
        x = x + self.pe[:, :x.size(1), :]
        return x
