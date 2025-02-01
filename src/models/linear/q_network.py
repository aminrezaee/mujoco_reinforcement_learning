from models.network_block_creator import create_network
from torch import nn
from entities.features import Run
from torch import Tensor, cat


class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        run = Run.instance()
        config = {
            "final_activation": None,
            "activation": run.network_config.activation_class,
            "hidden_layer_count": run.network_config.num_linear_layers,
            "shapes": run.network_config.linear_hidden_shapes
        }

        fully_connected_input_shape = int(run.network_config.input_shape * 2 +
                                          run.network_config.output_shape)
        self.first_network = create_network(config,
                                            input_shape=fully_connected_input_shape,
                                            output_shape=1,
                                            normalize_at_the_end=False,
                                            use_bias=run.network_config.use_bias,
                                            use_batchnorm=run.network_config.use_batch_norm)
        self.second_network = create_network(config,
                                             input_shape=fully_connected_input_shape,
                                             output_shape=1,
                                             normalize_at_the_end=False,
                                             use_bias=run.network_config.use_bias,
                                             use_batchnorm=run.network_config.use_batch_norm)

    def forward(self, state: Tensor, action: Tensor):
        input_tensor = cat([state.reshape(len(state), -1), action], 1)
        out1, out2 = self.first_network(input_tensor), self.second_network(input_tensor)
        # print(x.mean() , x.min() , x.max())
        return out1, out2
