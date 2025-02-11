from entities.features import Run

from torch.nn import Module, Embedding
import torch


class LearnedPositionalEncoding(Module):

    def __init__(self):
        super(LearnedPositionalEncoding, self).__init__()
        run = Run.instance()
        self.pos_embedding = Embedding(run.environment_config.window_length,
                                       run.network_config.feature_extractor_latent_size)

    def forward(self, x):
        """Add positional encoding to the input tensor x."""
        # x is expected to have shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
        position_embeddings = self.pos_embedding(positions)  # Shape: (1, seq_len, d_model)
        return x + position_embeddings
