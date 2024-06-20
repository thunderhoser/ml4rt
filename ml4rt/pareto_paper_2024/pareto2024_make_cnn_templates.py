"""Makes CNN templates for 2024 Pareto-front paper with Tom Beucler."""

import copy
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import cnn_architecture
from ml4rt.machine_learning import keras_losses as custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(
    scaling_factor=0.64
)

MODEL_DEPTHS = numpy.array([1, 2, 3, 4, 5, 6], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

DEFAULT_OPTION_DICT = {
    cnn_architecture.NUM_HEIGHTS_KEY: 127,
    cnn_architecture.NUM_INPUT_CHANNELS_KEY: 26,
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    cnn_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    cnn_architecture.L1_WEIGHT_KEY: 0.,
    cnn_architecture.L2_WEIGHT_KEY: 1e-7,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    cnn_architecture.USE_RESIDUAL_BLOCKS_KEY: False,
    # cnn_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    cnn_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    cnn_architecture.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    cnn_architecture.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION
}


def _run():
    """Makes CNN templates for 2024 Pareto-front paper with Tom Beucler.

    This is effectively the main method.
    """

    num_model_depths = len(MODEL_DEPTHS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    for i in range(num_model_depths):
        for j in range(num_channel_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            these_coeffs = numpy.logspace(
                0, MODEL_DEPTHS[i] - 1, num=MODEL_DEPTHS[i], base=2.
            )
            channel_counts = numpy.round(
                numpy.minimum(these_coeffs * FIRST_LAYER_CHANNEL_COUNTS[j], 64)
            ).astype(int)

            channel_counts = numpy.concatenate((
                channel_counts,
                numpy.array([1], dtype=int)
            ))

            option_dict[cnn_architecture.CONV_LAYER_CHANNEL_NUMS_KEY] = (
                channel_counts
            )
            option_dict[cnn_architecture.CONV_LAYER_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 0.)
            )
            option_dict[cnn_architecture.CONV_LAYER_FILTER_SIZES_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 3, dtype=int)
            )

            num_flattened_features = (
                127 *
                option_dict[cnn_architecture.CONV_LAYER_CHANNEL_NUMS_KEY][-1]
            )

            option_dict[cnn_architecture.DENSE_LAYER_NEURON_NUMS_KEY] = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=num_flattened_features,
                    num_classes=2, num_dense_layers=4, for_classification=False
                )[1]
            )

            model_object = cnn_architecture.create_model(option_dict)

            model_file_name = (
                'num-levels={0:d}_num-first-layer-channels={1:02d}/model.keras'
            ).format(
                MODEL_DEPTHS[i],
                FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )
            print('Writing model to: "{0:s}"...'.format(model_file_name))
            model_object.save(
                filepath=model_file_name, overwrite=True, include_optimizer=True
            )


if __name__ == '__main__':
    _run()
