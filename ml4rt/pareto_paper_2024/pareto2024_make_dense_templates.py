"""Makes dense-net templates for 2024 Pareto-front paper with Tom Beucler."""

import copy
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

MODEL_DEPTHS = numpy.array([1, 2, 3, 4, 5, 6], dtype=int)
HIDDEN_LAYER_NEURON_COUNTS = numpy.array(
    [64, 128, 256, 512, 1024, 2048], dtype=int
)

DEFAULT_OPTION_DICT = {
    dense_net_architecture.NUM_HEIGHTS_KEY: 127,
    dense_net_architecture.NUM_INPUT_CHANNELS_KEY: 26,
    dense_net_architecture.NUM_FLUX_COMPONENTS_KEY: 2,
    # dense_net_architecture.HIDDEN_LAYER_NEURON_NUMS_KEY: None,
    # dense_net_architecture.HIDDEN_LAYER_DROPOUT_RATES_KEY: None,
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

    num_model_depths = len(MODEL_DEPTHS)
    num_neuron_counts = len(HIDDEN_LAYER_NEURON_COUNTS)

    for i in range(num_model_depths):
        for j in range(num_neuron_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            option_dict[dense_net_architecture.HIDDEN_LAYER_NEURON_NUMS_KEY] = (
                numpy.full(
                    MODEL_DEPTHS[i], HIDDEN_LAYER_NEURON_COUNTS[j], dtype=int
                )
            )

            option_dict[
                dense_net_architecture.HIDDEN_LAYER_DROPOUT_RATES_KEY
            ] = numpy.full(MODEL_DEPTHS[i], 0.)

            model_object = dense_net_architecture.create_model(option_dict)

            model_file_name = (
                'num-levels={0:d}_num-neurons-per-layer={1:04d}/model.keras'
            ).format(
                MODEL_DEPTHS[i],
                HIDDEN_LAYER_NEURON_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )
            print('Writing model to: "{0:s}"...'.format( model_file_name
            ))
            model_object.save(
                filepath=model_file_name, overwrite=True, include_optimizer=True
            )


if __name__ == '__main__':
    _run()
