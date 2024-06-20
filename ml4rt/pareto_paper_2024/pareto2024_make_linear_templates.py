"""Makes dense-net templates for 2024 Pareto-front paper with Tom Beucler."""

import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import dense_net_architecture
from ml4rt.machine_learning import keras_losses as custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(
    scaling_factor=0.64
)

DEFAULT_OPTION_DICT = {
    dense_net_architecture.NUM_HEIGHTS_KEY: 127,
    dense_net_architecture.NUM_INPUT_CHANNELS_KEY: 26,
    dense_net_architecture.NUM_FLUX_COMPONENTS_KEY: 2,
    dense_net_architecture.HIDDEN_LAYER_NEURON_NUMS_KEY:
        numpy.array([], dtype=int),
    dense_net_architecture.HIDDEN_LAYER_DROPOUT_RATES_KEY:
        numpy.array([], dtype=float),
    dense_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    dense_net_architecture.HEATING_RATE_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.HEATING_RATE_ACTIV_FUNC_ALPHA_KEY: 0.,
    dense_net_architecture.FLUX_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.FLUX_ACTIV_FUNC_ALPHA_KEY: 0.,
    dense_net_architecture.L1_WEIGHT_KEY: 0.,
    dense_net_architecture.L2_WEIGHT_KEY: 1e-7,
    dense_net_architecture.USE_BATCH_NORM_KEY: True,
    dense_net_architecture.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    dense_net_architecture.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION
}


def _run():
    """Makes dense-net templates for 2024 Pareto-front paper with Tom Beucler.

    This is effectively the main method.
    """

    model_object = dense_net_architecture.create_model(DEFAULT_OPTION_DICT)

    model_file_name = 'linear_regression/model.keras'

    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)
    print('Writing model to: "{0:s}"...'.format(model_file_name))
    model_object.save(
        filepath=model_file_name, overwrite=True, include_optimizer=True
    )


if __name__ == '__main__':
    _run()
